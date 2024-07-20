from agent_super import TradingAgent
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class DecisionTreeAgent(TradingAgent):
    def __init__(self, short_window=100, initial_cash=100000):
        super().__init__(initial_cash)
        self.short_window = short_window  # Use the last 100 observations

    def train_model(self, data):
        data = data.copy()
        data['Future_Close'] = data['Close'].shift(-1)
        data['Target'] = np.where(data['Future_Close'] > data['Close'], 1, 2)
        data.dropna(inplace=True)

        # Use the last 100 observations for training
        if len(data) < self.short_window:
            return False  # Not enough data to train the model

        features = data[['Close']].iloc[-self.short_window:].values
        target = data['Target'].iloc[-self.short_window:].values

        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(features, target)
        return True  # Model training was successful

    def generate_signals(self, data):
        if not self.train_model(data):  # Retrain the model each time a signal is generated
            return 0  # Hold signal if there aren't enough observations to train the model

        data = data.copy()
        features = data[['Close']].iloc[-self.short_window:].values[-1].reshape(1, -1)
        
        prediction = self.model.predict(features)
        print(prediction)
        if prediction == 1:
            return 1  # Buy signal
        else:
            return 2  # Sell signal