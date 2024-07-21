from agent_super import TradingAgent
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
class RandomForestAgent(TradingAgent):
    def __init__(self, short_window=50, long_window=200, initial_cash=100000):
        super().__init__(initial_cash)
        self.name="RF"
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate_rsi(self, data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data, fast_period=13, slow_period=26, signal_period=9):
        fast_ema = data['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['Close'].ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal_line

    def calculate_technical_indicators(self, data):
        data.loc[:, 'SMA_20'] = data['Close'].rolling(window=self.short_window).mean()
        data.loc[:, 'SMA_50'] = data['Close'].rolling(window=self.long_window).mean()
        data.loc[:, 'RSI'] = self.calculate_rsi(data)
        data.loc[:, 'MACD'], data.loc[:, 'Signal_Line'] = self.calculate_macd(data)
        #data.dropna(inplace=True)
        return data

    def train_model(self, data):
        data=data.copy()
        data = self.calculate_technical_indicators(data)
        data['Future_Close'] = data['Close'].shift(-1)
        data['Target'] = np.where(data['Future_Close'] > data['Close'], 1, 2)
        data.dropna(inplace=True)
        features = data[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']]
        target = data['Target']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")

    def generate_signals(self, data):
        data=data.copy()
        data = self.calculate_technical_indicators(data)
        features = data[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']].iloc[-1].values
        
        # Check if any value in features is NaN
        if np.isnan(features).any():
            return 0  # Hold signal

        features = features.reshape(1, -1)
        prediction = self.model.predict(features)
        print(prediction)
        if prediction == 1:
            return 1  # Buy signal
        else:
            return 2  # Sell signal