from agent_super import TradingAgent
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class HMMAgent(TradingAgent):
    def __init__(self, short_window=50, long_window=200, initial_cash=100000, n_components=4):
        super().__init__(initial_cash)
        self.short_window = short_window
        self.long_window = long_window
        self.n_components = n_components  # Number of HMM components
        self.hmm = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
    
    def calculate_rsi(self, data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = data['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['Close'].ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal_line

    def calculate_technical_indicators(self, data):
        data['SMA_20'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA_50'] = data['Close'].rolling(window=self.long_window).mean()
        data['RSI'] = self.calculate_rsi(data)
        data['MACD'], data['Signal_Line'] = self.calculate_macd(data)
        #data.dropna(inplace=True)
        return data

    def train_model(self, data):
        data = data.copy()
        data = self.calculate_technical_indicators(data)
        data['Future_Close'] = data['Close'].shift(-1)
        data['Target'] = np.where(data['Future_Close'] > data['Close'], 1, 2)
        data.dropna(inplace=True)
        
        # Features for HMM: SMA_20, SMA_50, RSI, MACD, Signal_Line
        features = data[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']].values

        # Train HMM
        self.hmm.fit(features)

        # Predict hidden states
        hidden_states = self.hmm.predict(features)

        # Align the target variable with the hidden states
        data['Hidden_State'] = hidden_states
        target = data['Target'].values

        # Use hidden states as features for a secondary model (Random Forest)
        X_train, X_test, y_train, y_test = train_test_split(hidden_states.reshape(-1, 1), target, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")

    def generate_signals(self, data):
        data = data.copy()
        data = self.calculate_technical_indicators(data)
        
        # Features for HMM: SMA_20, SMA_50, RSI, MACD, Signal_Line
        features = data[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']].values[-self.short_window:]

        if len(features) < self.short_window or np.isnan(features).any():
            return 0  # Not enough data for the rolling window

        # Predict hidden states
        hidden_states = self.hmm.predict(features)

        # Use the latest hidden state to generate a signal
        last_hidden_state = hidden_states[-1].reshape(1, -1)
        prediction = self.model.predict(last_hidden_state)
        print(prediction)
        if prediction == 1:
            return 1  # Buy signal
        else:
            return 2  # Sell signal