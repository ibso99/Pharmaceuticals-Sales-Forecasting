import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

class LSTMSalesModel:
    def __init__(self, filepath, lag=5):
        self.filepath = filepath
        self.lag = lag
        self.data = None
        self.scaler = None
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.filepath)
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df = df.set_index('Date').sort_index()
        self.data = df[['Sales']]

    def check_stationarity(self):
        result = adfuller(self.data['Sales'])
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        for key, value in result[4].items():
            print(f'Critical Value ({key}): {value}')
        return result[1] < 0.05

    def difference_data(self):
        self.data['Sales'] = self.data['Sales'].diff().dropna()

    def plot_correlations(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_acf(self.data['Sales'], ax=axes[0])
        plot_pacf(self.data['Sales'], ax=axes[1])
        plt.show()

    def create_supervised_data(self):
        data = pd.DataFrame(self.data['Sales'])
        for i in range(1, self.lag + 1):
            data[f'lag_{i}'] = data['Sales'].shift(i)
        data = data.dropna()
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values
        return X, y

    def scale_data(self, X, y):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
        return X_scaled, y_scaled

    def build_lstm_model(self, input_shape):
        self.model = Sequential([
            LSTM(50, activation='tanh', input_shape=input_shape, return_sequences=True),
            LSTM(50, activation='tanh'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stop]
        )
        return history

    def run_pipeline(self, save_path='lstm_sales_model.h5'):
        # Load data
        self.load_data()

        # Check stationarity
        is_stationarity = self.check_stationarity()
        if not is_stationarity:
            self.difference_data()

        # Plot autocorrelation and partial autocorrelation
        self.plot_correlations()

        # Create supervised learning data
        X, y = self.create_supervised_data()

        # Scale data
        X_scaled, y_scaled = self.scale_data(X, y)

        # Reshape X for LSTM
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        # Build and train the model
        self.build_lstm_model((X_scaled.shape[1], 1))
        history = self.train_model(X_train, y_train, X_val, y_val)

        # Plot training loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

        # Save the model
        self.model.save(save_path)
        print(f"Model saved at '{save_path}'")

