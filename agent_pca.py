from agent_super import TradingAgent
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class PCAAgent(TradingAgent):
    def __init__(self, short_window=50, long_window=200, initial_cash=100000, n_components=3):
        super().__init__(initial_cash)
        self.name="PCA"
        self.short_window = short_window
        self.long_window = long_window
        self.n_components = n_components  # Number of PCA components
        self.pca = PCA(n_components=n_components)

    def calculate_technical_indicators(self, data):
        # No technical indicators calculation, just use the raw 'Close' prices for PCA
        return data

    def train_model(self, data):
        data = data.copy()
        data = self.calculate_technical_indicators(data)
        data['Future_Close'] = data['Close'].shift(-1)
        data['Target'] = np.where(data['Future_Close'] > data['Close'], 1, 2)
        data.dropna(inplace=True)

        # Use the 'Close' prices directly for PCA
        features = data[['Close']].values

        # Create a rolling window of features to apply PCA
        rolling_features = np.array([features[i:i+self.short_window] for i in range(len(features)-self.short_window+1)])
        rolling_features = rolling_features.reshape(rolling_features.shape[0], -1)

        # Apply PCA on rolling window features
        pca_features = self.pca.fit_transform(rolling_features)

        # Align the target variable with the transformed features
        target = data['Target'].iloc[self.short_window-1:].values

        X_train, X_test, y_train, y_test = train_test_split(pca_features, target, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")

    def generate_signals(self, data):
        data = data.copy()
        data = self.calculate_technical_indicators(data)
        
        # Use the 'Close' prices directly for PCA
        features = data[['Close']].values[-self.short_window:]

        # Create a rolling window of features to apply PCA
        if len(features) < self.short_window:
            return 0  # Not enough data for the rolling window

        rolling_features = features.reshape(1, -1)

        # Apply PCA on rolling window features
        pca_features = self.pca.transform(rolling_features)

        prediction = self.model.predict(pca_features)
        print(prediction)
        if prediction == 1:
            return 1  # Buy signal
        else:
            return 2  # Sell signal