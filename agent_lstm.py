from agent_super import TradingAgent
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

class LSTMAgent(TradingAgent):
    def __init__(self, time_step=50, initial_cash=100000):
        super().__init__(initial_cash)
        self.time_step = time_step
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(self.time_step, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def train_model(self, data):
        data = data.copy()
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.time_step, len(scaled_data)):
            X.append(scaled_data[i-self.time_step:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2)
    
    def generate_signals(self, data):
        data = data.copy()
        scaled_data = self.scaler.transform(data['Close'].values.reshape(-1, 1))

        if len(scaled_data) < self.time_step:
            return 0  # Not enough data to make a prediction

        X_test = []
        for i in range(self.time_step, len(scaled_data)):
            X_test.append(scaled_data[i-self.time_step:i, 0])
        
        if len(X_test) == 0:
            return 0  # No valid test data

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)

        current_price = data['Close'].values[-1]
        predicted_price = predictions[-1, 0]

        if predicted_price > current_price:
            return 1  # Buy signal
        elif predicted_price < current_price:
            return 2  # Sell signal
        return 0  # Hold signal