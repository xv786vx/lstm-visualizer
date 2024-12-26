import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def fetch_stock_data(ticker, start_date, end_date):

    # HELPER FUNCS
    def calc_rsi(stock, window=14):
        delta = stock['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calc_ema(stock, window=20):
        return stock['Close'].ewm(span=window, adjust=False).mean()

    def calc_vwap(stock):
        return (((stock['High'] + stock['Low'] + stock['Close']) / 3) * stock['Volume']).cumsum() / stock['Volume'].cumsum()
    

    stock = yf.download(ticker, start=start_date, end=end_date)
    stock.columns = stock.columns.get_level_values(0)

    # feature engineering
    stock['Daily Return'] = stock['Close'].pct_change()
    stock['50MA'] = stock['Close'].rolling(window=50).mean()
    stock['200MA'] = stock['Close'].rolling(window=200).mean()
    stock['Volatility'] = stock['Daily Return'].rolling(window=10).std()
    stock['RSI'] = calc_rsi(stock)
    stock['VWAP'] = calc_vwap(stock)
    stock['20EMA'] = calc_ema(stock)
    stock = stock.dropna()

    features = ['Close', 'Daily Return', '50MA', '200MA', 'Volatility', 'RSI', 'VWAP', '20EMA']
    X = stock[features]
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(X), X
     

def create_sequences(data, seq_length):
    X, y = [],[]
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :]) # captures all the features in length: seq_length
        y.append(data[i + seq_length, 0]) # captures target: next day's 'Close'
    X, y = np.array(X), np.array(y)
    return X, y

def build_model(X, y, units, epochs_num):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=units),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=epochs_num, batch_size=32, verbose=0)
    print("Model built and trained!")
    return model, X_test, y_test

def predict(model, X_test, y_test, X, features=['Close', 'Daily Return', '50MA', '200MA', 'Volatility', 'RSI', 'VWAP', '20EMA']):
    predicted_prices = model.predict(X_test)
    scaler = MinMaxScaler(feature_range=(0, 1))
    obj = scaler.fit(X)
    
    # inverse transform y_test (actual prices) and predicted_prices
    temp = np.zeros((predicted_prices.shape[0], len(features))) # 8 is len(features)
    temp[:,0] = predicted_prices[:,0] 
    predicted_prices_actual = obj.inverse_transform(temp)[:,0] 
    print("predicted_prices_actual inverted")

    # doing the same for y_test
    temp = np.zeros((y_test.shape[0], len(features))) # 8 is len(features)
    temp[:, 0] = y_test[:, 0] if y_test.ndim > 1 else y_test 
    y_test_actual = obj.inverse_transform(temp)[:, 0] 
    print("y_test_actual inverted")

    return predicted_prices_actual, y_test_actual

def plot_results(ticker, actual, predicted, filename='plot.png'):
    plt.plot(actual, label='Actual Prices', color='blue')
    plt.plot(predicted, label='Predicted Prices', color='orange')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join('static', filename)
    plt.savefig(plot_path)
    plt.close()
    return filename

def lstm_strategy(ticker, start_date, end_date, seq_length, units, epochs_num):

    data, X = fetch_stock_data(ticker, start_date, end_date)
    X_seq, y_seq = create_sequences(data, seq_length)
    model, X_test, y_test = build_model(X_seq, y_seq, units, epochs_num)
    predicted_prices, actual_prices = predict(model, X_test, y_test, X)
    plot_path = plot_results(ticker, actual_prices, predicted_prices)
    print(plot_path)
    return plot_path