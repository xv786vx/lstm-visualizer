import sys
import os
from tkinter import TOP
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

TOP_50_SP500 = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'NFLX', 
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
            'CRM', 'INTC', 'VZ', 'CMCSA', 'PFE', 'ABT', 'KO', 'PEP', 'TMO', 
            'COST', 'WMT', 'MRK', 'BAC', 'XOM', 'CVX', 'LLY', 'ABBV', 'ORCL',
            'WFC', 'BMY', 'MDT', 'ACN', 'DHR', 'TXN', 'QCOM', 'HON', 'IBM',
            'AMGN', 'UPS', 'LOW', 'SBUX', 'CAT'
        ]

import yfinance as yf
from lstm_strategy_v2 import LSTMStockPredictor, StockDataFetcher
predictor = LSTMStockPredictor("../backend/models/test_lstm.keras", seq_length=30)

TOP_50_SP500 = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'NFLX', 
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
            'CRM', 'INTC', 'VZ', 'CMCSA', 'PFE', 'ABT', 'KO', 'PEP', 'TMO', 
            'COST', 'WMT', 'MRK', 'BAC', 'XOM', 'CVX', 'LLY', 'ABBV', 'ORCL',
            'WFC', 'BMY', 'MDT', 'ACN', 'DHR', 'TXN', 'QCOM', 'HON', 'IBM',
            'AMGN', 'UPS', 'LOW', 'SBUX', 'CAT'
        ]

TOP_10_SP500 = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'NFLX', 
            'JPM', 'JNJ'
        ]

# step 1: Fetch data
print("\n\nSTEP 1: FETCHING DATA")
data = predictor.fetch_and_prepare_data(TOP_10_SP500, '2020-01-01', '2024-01-01')
print("dimensions:", data.shape)
print("First 5 rows of data:")
print(data.head().to_string(index=False, float_format='%.3f', max_cols=None))
print("Last 5 rows of data:")
print(data.tail().to_string(index=False, float_format='%.3f', max_cols=None))

# step 2: normalize data
print("\n\nSTEP 2: NORMALIZING DATA")
normalized_data = predictor.normalize_data(data)
print("First 5 rows of normalized data:")
print(normalized_data.head().to_string(index=False, float_format='%.3f', max_cols=None))
print("Last 5 rows of normalized data:")
print(normalized_data.tail().to_string(index=False, float_format='%.3f', max_cols=None))

# step 3: train model
print("\n\nSTEP 3: TRAINING MODEL")
X, y, _ = predictor.create_sequences(normalized_data, TOP_10_SP500)