from agent_super import TradingAgent
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data, window_size):
        super(TradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.current_step = window_size
        self.initial_cash = 100000
        self.cash = self.initial_cash
        self.shares = 0
        self.net_worth = self.initial_cash
        self.total_profit = 0

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(window_size, len(data.columns)), dtype=np.float16)

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares = 0
        self.net_worth = self.initial_cash
        self.total_profit = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data.iloc[self.current_step-self.window_size:self.current_step].values
        return obs

    def step(self, action):
        current_price = self.data['Close'].iloc[self.current_step]
        reward = 0

        if action == 1:  # Buy
            if self.cash > 0:
                self.shares = self.cash // current_price
                self.cash -= self.shares * current_price
        elif action == 2:  # Sell
            if self.shares > 0:
                self.cash += self.shares * current_price
                self.shares = 0

        self.current_step += 1
        self.net_worth = self.cash + self.shares * current_price
        self.total_profit = self.net_worth - self.initial_cash

        reward = self.total_profit

        done = self.current_step == len(self.data) - 1

        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass


class RLAgent(TradingAgent):
    def __init__(self, window_size=50, initial_cash=100000):
        super().__init__(initial_cash)
        self.window_size = window_size
        self.model = None

    def train_model(self, data):
        data = data.copy()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        env = DummyVecEnv([lambda: TradingEnv(scaled_data, self.window_size)])
        self.model = DQN('MlpPolicy', env, verbose=1)
        self.model.learn(total_timesteps=10000)

    def generate_signals(self, data):
        data = data.copy()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        env = DummyVecEnv([lambda: TradingEnv(scaled_data, self.window_size)])
        obs = env.reset()

        for _ in range(len(scaled_data) - self.window_size):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if dones:
                break

        if action == 1:
            return 1  # Buy signal
        elif action == 2:
            return 2  # Sell signal
        return 0  # Hold signal

# Example usage:
# agent = RLAgent()
# agent.train_model(historical_data)
# signal = agent.generate_signals(new_data)