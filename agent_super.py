import numpy as np
import pandas as pd

class TradingAgent:
    def __init__(self,init_cash):
        self.name = "Dummy"
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.cash = init_cash  # Starting cash in USD
        self.holdings = 0

    def generate_signals(self):
        # Randomly choose an action: 0 (Hold), 1 (Buy), 2 (Sell)
        return np.random.choice([0, 1, 2])
    
    def train_model(self, data):
        pass

    def trade(self, data):
        signal = self.generate_signals(data)
        price = data['Close'].iloc[-1]
        if signal == 1 and self.position != 1:
            if self.cash > 0:
                self.holdings = self.cash / price
                self.cash = 0
                self.position = 1
                print(f"{pd.Timestamp.now()}: {self.name} Buy at {price}")
        elif signal == 2 and self.position != -1:
            if self.holdings > 0:
                self.cash = self.holdings * price
                self.holdings = 0
                self.position = -1
                print(f"{pd.Timestamp.now()}: {self.name} Sell at {price}")
        else:
            print(f"{pd.Timestamp.now()}: {self.name} Hold")

    def get_portfolio_value(self, current_price):
        return self.cash + (self.holdings * current_price)