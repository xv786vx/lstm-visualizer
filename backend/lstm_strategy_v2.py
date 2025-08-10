import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import joblib
from joblib import dump, load
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, r2_score

"""## Strategy #1 Part 1: Preprocessing data

"""

class StockDataFetcher:
    """Fetches stock data and creates features for model training"""
    def calc_technical_indicators(stock_data):
        """Calculate technical indicators for a stock DataFrame"""
        # RSI
        delta = stock_data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))

        # Other indicators
        stock_data['Daily Return'] = stock_data['Close'].pct_change()
        stock_data['50MA'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['200MA'] = stock_data['Close'].rolling(window=200).mean()
        stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=10).std()
        stock_data['20EMA'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
        stock_data['VWAP'] = (((stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3)
                             * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()

        return stock_data

class LSTMStockPredictor:
    """Main class for stock prediction using LSTM, Dense, Regularization layers"""
    def __init__(self, model_path, seq_length):
        self.model_path = model_path
        self.seq_length = seq_length
        self.scalers = {}
        self.model = None
        self.features = ['Close', 'Daily Return', '50MA', '200MA',
                        'Volatility', 'RSI', 'VWAP', '20EMA']
        self.training_tickers = None
        self.model_metadata_path = model_path.replace('.keras', '_metadata.json')
        
        # Fixed list of top 50 S&P 500 stocks for consistent input dimensions
        self.TOP_50_SP500 = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'NFLX', 
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
            'CRM', 'INTC', 'VZ', 'CMCSA', 'PFE', 'ABT', 'KO', 'PEP', 'TMO', 
            'COST', 'WMT', 'MRK', 'BAC', 'XOM', 'CVX', 'LLY', 'ABBV', 'ORCL',
            'WFC', 'BMY', 'MDT', 'ACN', 'DHR', 'TXN', 'QCOM', 'HON', 'IBM',
            'AMGN', 'UPS', 'LOW', 'SBUX', 'CAT'
        ]

        # Load model and check if it matches current tickers
        self._load_model_if_compatible()

    def _load_model_if_compatible(self):
        """Load model only if it exists and was trained on compatible tickers"""
        if os.path.exists(self.model_path) and os.path.exists(self.model_metadata_path):
            try:
                import json
                with open(self.model_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.training_tickers = metadata.get('training_tickers', [])
                    self.training_start_date = metadata.get('training_start_date')
                    self.training_end_date = metadata.get('training_end_date')
                    print(f"Found existing model trained on: {self.training_tickers}")
                    print(f"Training date range: {self.training_start_date} to {self.training_end_date}")
                    
                self.model = load_model(self.model_path)
                print(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model metadata: {e}")
                self._clear_model()
        else:
            self.model = None
            self.training_tickers = None
            self.training_start_date = None
            self.training_end_date = None

    def _clear_model(self):
        """Clear existing model and metadata files"""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            print(f"Removed incompatible model: {self.model_path}")
        if os.path.exists(self.model_metadata_path):
            os.remove(self.model_metadata_path)
            print(f"Removed model metadata: {self.model_metadata_path}")
        self.model = None
        self.training_tickers = None
        self.training_start_date = None
        self.training_end_date = None

    def _save_model_metadata(self, tickers, start_date=None, end_date=None):
        """Save metadata about the model including training tickers and date range"""
        import json
        metadata = {
            'training_tickers': sorted(tickers),  # Sort for consistent comparison
            'seq_length': self.seq_length,
            'features': self.features,
            'training_start_date': start_date,
            'training_end_date': end_date,
            'created_at': str(pd.Timestamp.now())
        }
        with open(self.model_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved model metadata to {self.model_metadata_path}")

    def _tickers_match(self, new_tickers):
        """Check if new tickers are a subset of the 50 stocks the model was trained on"""
        if self.training_tickers is None:
            return False
        # For the fixed 50-stock model, check if it was trained on all 50 stocks
        return sorted(self.training_tickers) == sorted(self.TOP_50_SP500)

    def _date_range_compatible(self, start_date, end_date):
        """Check if requested date range is within the model's training date range"""
        if not self.training_start_date or not self.training_end_date:
            return False
        
        try:
            train_start = pd.Timestamp(self.training_start_date)
            train_end = pd.Timestamp(self.training_end_date)
            req_start = pd.Timestamp(start_date)
            req_end = pd.Timestamp(end_date)
            
            # Requested range must be within training range
            return req_start >= train_start and req_end <= train_end
        except Exception as e:
            print(f"Error validating date range: {e}")
            return False

    def can_predict_without_training(self, tickers, start_date, end_date):
        """Check if we can make predictions without retraining"""
        if self.model is None:
            return False
        
        # Check if all requested tickers are in training set
        if not all(ticker in self.TOP_50_SP500 for ticker in tickers):
            return False
        
        # Check if model was trained on all 50 stocks
        if not self._tickers_match(self.TOP_50_SP500):
            return False
            
        # Check if date range is compatible
        return self._date_range_compatible(start_date, end_date)


    def fetch_and_prepare_data(self, tickers, start_date, end_date):
        """Fetch and prepare stock data for multiple tickers"""
        all_data = []
        for ticker in tickers:
            # Fetch data
            stock = yf.download(ticker, start=start_date, end=end_date)
            stock.columns = stock.columns.get_level_values(0)  # Remove multi-level columns
            stock['Ticker'] = ticker

            # Calculate indicators
            stock = StockDataFetcher.calc_technical_indicators(stock)
            stock.dropna(inplace=True)
            all_data.append(stock)

        combined_data = pd.concat(all_data, axis=0)
        return combined_data.reset_index()


    def normalize_data(self, data):
        """Normalize data by stock"""
        normalized_data = data.copy()  # Create a copy to avoid modifying original data

        # this does NOT normalize the following; "High", "Low", "Open", "Volume"
        for ticker, group in normalized_data.groupby('Ticker'):
            if ticker not in self.scalers:
                print(f"Creating new scaler for {ticker}")
                scaler = StandardScaler()
                # Fit the scaler on the features
                scaler.fit(group[self.features])
                self.scalers[ticker] = scaler
            else:
                print(f"Using existing scaler for {ticker}")

            # Transform the data
            normalized_values = self.scalers[ticker].transform(group[self.features])
            normalized_data.loc[group.index, self.features] = normalized_values

        return normalized_data

    def train_model_on_all_50_stocks(self, start_date='2020-01-01', end_date='2025-01-01'):
        """One-time training on all 50 S&P stocks for fixed input dimensions"""
        print("Starting training on all 50 S&P stocks...")
        print(f"Date range: {start_date} to {end_date}")
        
        # Store current dates for metadata
        self._current_start_date = start_date
        self._current_end_date = end_date
        
        # Check if we can skip training by using existing model
        if self.can_predict_without_training(self.TOP_50_SP500, start_date, end_date):
            print("Using existing pre-trained model - skipping training!")
            return None, None  # Return dummy values since model is already loaded
        
        print("Existing model not compatible - proceeding with training...")
        
        # Fetch data for all 50 stocks
        try:
            data = self.fetch_and_prepare_data(self.TOP_50_SP500, start_date, end_date)
            print(f"Successfully fetched data for all 50 stocks")
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
        
        # Normalize data
        normalized_data = self.normalize_data(data)
        print("Data normalized successfully")
        
        # Create sequences with fixed 50-stock dimensions
        X, y, _ = self.create_sequences(normalized_data, self.TOP_50_SP500)
        print(f"Created {len(X)} sequences with shape {X.shape}")
        
        if len(X) == 0:
            print("Error: No sequences created")
            return False
        
        # Train the model
        X_test, y_test = self.train_save_model(X, y, self.TOP_50_SP500)
        print("Model training completed successfully")
        
        # Evaluate model performance
        if len(X_test) > 0:
            test_predictions = self.model.predict(X_test, verbose=0)
            mse = mean_squared_error(y_test, test_predictions)
            print(f"Test MSE: {mse:.4f}")
        
        return X_test, y_test

    def create_sequences(self, data, selected_tickers, original_close=None):
        """Create sequences with fixed 50-stock dimensions for consistent input shape"""
        X, y, dates = [], [], []

        # Always use the fixed 50 S&P stocks for consistent one-hot encoding
        ticker_to_idx = {t: idx for idx, t in enumerate(self.TOP_50_SP500)}
        one_hot_size = 50  # Always 50, regardless of selection

        for ticker in selected_tickers:  # Only process selected stocks
            if ticker not in self.TOP_50_SP500:
                print(f"Warning: {ticker} not in TOP_50_SP500 list, skipping...")
                continue
                
            ticker_data = data[data['Ticker'] == ticker]
            if len(ticker_data) == 0:
                print(f"Warning: No data found for {ticker}, skipping...")
                continue
                
            group = ticker_data
            values = group[self.features].values

            for i in range(len(values) - self.seq_length):
                sequence = values[i:i + self.seq_length]  # Shape: [30, 8]
                
                # Handle target price correctly based on original_close type
                if original_close is not None and isinstance(original_close, dict):
                    # Use dictionary format
                    target = original_close[ticker].iloc[i + self.seq_length]
                elif original_close is not None:
                    # Handle MultiIndex Series (legacy format)
                    ticker_close = original_close.reset_index()
                    ticker_close = ticker_close[ticker_close['Ticker'] == ticker].set_index('Date')['Close']
                    target = ticker_close.iloc[i + self.seq_length]
                else:
                    target = group['Close'].iloc[i + self.seq_length]
                    
                dates.append(group.index[i + self.seq_length])

                # Create one-hot encoding (always size 50)
                ticker_one_hot = np.zeros(one_hot_size)
                ticker_one_hot[ticker_to_idx[ticker]] = 1

                # Stack features with one-hot encoding: [30, 8] + [30, 50] = [30, 58]
                sequence_with_ticker = np.column_stack([sequence, np.tile(ticker_one_hot, (self.seq_length, 1))])
                X.append(sequence_with_ticker)
                y.append(target)

        return np.array(X), np.array(y), pd.DatetimeIndex(dates)

    def train_save_model(self, X, y, tickers):
      """Train and save the LSTM model"""
      print("\nChecking model compatibility...")
      
      # For 50-stock fixed model, check if trained on all 50 stocks
      is_50_stock_model = self._tickers_match(tickers)
      
      # Check if we need to retrain
      if not is_50_stock_model:
          if self.training_tickers is not None:
              print(f"Model not compatible with 50-stock architecture. Retraining...")
              self._clear_model()
          else:
              print("No existing compatible model found.")
      
      # Split data regardless of whether we train or not
      split_idx = int(len(X) * 0.8)
      X_train, X_test = X[:split_idx], X[split_idx:]
      y_train, y_test = y[:split_idx], y[split_idx:]

      if self.model is not None and is_50_stock_model:
          print(f"Using existing 50-stock model")
      else:
          print("Training new 50-stock model...")
          print(f"Input shape: {X.shape}")
          print(f"Expected input shape: [batch_size, {self.seq_length}, 58] (8 features + 50 one-hot)")

          # Fixed input shape: [batch_size, seq_length, 58] (8 features + 50 one-hot)
          self.model = Sequential([
              LSTM(64, return_sequences=True, input_shape=(self.seq_length, 58)),
              Dropout(0.2),
              LSTM(32, return_sequences=False),
              Dropout(0.2),
              Dense(16, activation='relu'),
              Dense(1)
          ])
          self.model.compile(optimizer='adam', loss='mse')

          # Train model
          self.model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
          self.model.save(self.model_path)
          self.training_tickers = sorted(self.TOP_50_SP500)  # Always trained on all 50
          self._save_model_metadata(self.TOP_50_SP500, 
                                   getattr(self, '_current_start_date', None),
                                   getattr(self, '_current_end_date', None))
          print(f"Model saved to {self.model_path}")

      return X_test, y_test


    def backtest_stock(self, ticker, start_date, end_date, selected_tickers=None):
        """Perform backtesting for a single ticker using fixed 50-stock input"""
        print(f"\nBacktesting {ticker}...")

        if ticker not in self.TOP_50_SP500:
            print(f"Error: {ticker} not in TOP_50_SP500 list")
            return None

        # Get and prepare data
        data = self.fetch_and_prepare_data([ticker], start_date, end_date)
        # Store original close prices in dictionary format for create_sequences
        original_close_dict = {ticker: data.set_index('Date')['Close']}
        normalized_data = self.normalize_data(data)

        # Create sequences using fixed 50-stock dimensions (only for selected ticker)
        X, y, dates = self.create_sequences(normalized_data, [ticker], original_close_dict)

        if len(X) < self.seq_length + 32:
            print(f"Not enough data for {ticker}")
            return None

        # Split data for testing
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        test_dates = dates[split_idx:]

        print(f"Input shape: {X_test.shape}")  # Should be [batch, 30, 58]

        # Make predictions
        predictions = self.model.predict(X_test, batch_size=32, verbose=0).flatten()

        dummy_array = np.zeros((len(predictions), len(self.features)))
        # Put predictions in the 'Close' price column position
        dummy_array[:, 0] = predictions
        # Inverse transform
        predictions = self.scalers[ticker].inverse_transform(dummy_array)[:, 0]

        # Create results DataFrame with portfolio management
        backtest_df = pd.DataFrame({
            'Date': test_dates,
            'Close': y_test,
            'Predicted': predictions,
            'Position': 0,
            'Cash': 10000,  # Initial cash
            'Holdings': 0,
            'Total_Value': 10000
        }).set_index('Date')

        # Generate trading signals and manage portfolio
        for i in range(1, len(backtest_df)):
            prev_cash = backtest_df['Cash'].iloc[i-1]
            prev_holdings = backtest_df['Holdings'].iloc[i-1]
            current_price = backtest_df['Close'].iloc[i]
            predicted_price = backtest_df['Predicted'].iloc[i]

            # Generate signal based on predicted price movement
            signal = 1 if predicted_price > current_price else 0

            # Execute trades based on signal
            if signal == 1 and prev_cash >= current_price:
                # Buy signal - invest all available cash
                shares_to_buy = prev_cash // current_price
                new_holdings = prev_holdings + shares_to_buy
                new_cash = prev_cash - (shares_to_buy * current_price)
            elif signal == 0 and prev_holdings > 0:
                # Sell signal - sell all holdings
                new_cash = prev_cash + (prev_holdings * current_price)
                new_holdings = 0
            else:
                # Hold current position
                new_cash = prev_cash
                new_holdings = prev_holdings

            # Calculate total portfolio value
            total_value = new_cash + (new_holdings * current_price)

            # Update DataFrame
            backtest_df.iloc[i, backtest_df.columns.get_loc('Cash')] = new_cash
            backtest_df.iloc[i, backtest_df.columns.get_loc('Holdings')] = new_holdings
            backtest_df.iloc[i, backtest_df.columns.get_loc('Total_Value')] = total_value
            backtest_df.iloc[i, backtest_df.columns.get_loc('Position')] = signal

        return backtest_df

    def predict_multiple_stocks(self, selected_tickers, start_date, end_date):
        """Make predictions for multiple selected stocks using the 50-stock model"""
        print(f"\nMaking predictions for {len(selected_tickers)} stocks: {selected_tickers}")
        
        # Validate selected tickers
        valid_tickers = [ticker for ticker in selected_tickers if ticker in self.TOP_50_SP500]
        if len(valid_tickers) != len(selected_tickers):
            invalid_tickers = [ticker for ticker in selected_tickers if ticker not in self.TOP_50_SP500]
            print(f"Warning: Invalid tickers (not in TOP_50_SP500): {invalid_tickers}")
        
        if len(valid_tickers) == 0:
            print("Error: No valid tickers provided")
            return None
            
        # Fetch and prepare data for selected stocks
        data = self.fetch_and_prepare_data(valid_tickers, start_date, end_date)
        normalized_data = self.normalize_data(data)
        
        # Create sequences with fixed 50-stock dimensions
        X, y, dates = self.create_sequences(normalized_data, valid_tickers)
        
        if len(X) == 0:
            print("Error: No sequences created")
            return None
            
        print(f"Created {len(X)} sequences with shape {X.shape}")
        
        # Make predictions
        predictions = self.model.predict(X, batch_size=32, verbose=0).flatten()
        
        # Group predictions by ticker
        results = {}
        start_idx = 0
        
        for ticker in valid_tickers:
            ticker_data = data[data['Ticker'] == ticker]
            num_sequences = len(ticker_data) - self.seq_length
            
            if num_sequences > 0:
                ticker_predictions = predictions[start_idx:start_idx + num_sequences]
                
                # Inverse transform predictions
                dummy_array = np.zeros((len(ticker_predictions), len(self.features)))
                dummy_array[:, 0] = ticker_predictions
                ticker_predictions_scaled = self.scalers[ticker].inverse_transform(dummy_array)[:, 0]
                
                results[ticker] = {
                    'predictions': ticker_predictions_scaled.tolist(),
                    'dates': dates[start_idx:start_idx + num_sequences].tolist(),
                    'actual_prices': y[start_idx:start_idx + num_sequences].tolist()
                }
                
                start_idx += num_sequences
        
        return results


    def plot_results(self, backtest_df, ticker, start_date, end_date, save_path=None):
      """Plot backtest results including price predictions and portfolio value"""
      # Set dark background style for better visibility
      plt.style.use('dark_background')
      fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
      
      # Set figure background to transparent
      fig.patch.set_alpha(0.0)

      # Plot 1: Price and Predictions
      ax1.plot(backtest_df.index, backtest_df['Close'],
              label='Actual', color='white', linewidth=2)
      ax1.plot(backtest_df.index, backtest_df['Predicted'],
              label='Predicted', color='#ff6b6b', linestyle='--', linewidth=2)
      ax1.set_title(f'{ticker} Price Prediction', color='white', fontsize=16, fontweight='bold')
      ax1.set_xlabel('Date', color='white', fontsize=12)
      ax1.set_ylabel('Price ($)', color='white', fontsize=12)
      ax1.legend(fontsize=12)
      ax1.grid(True, alpha=0.3)
      ax1.tick_params(colors='white')
      ax1.spines['bottom'].set_color('white')
      ax1.spines['top'].set_color('white')
      ax1.spines['right'].set_color('white')
      ax1.spines['left'].set_color('white')

      # Plot 2: Portfolio Value and Returns
      portfolio_value = backtest_df['Total_Value']
      market_value = backtest_df['Close'] / backtest_df['Close'].iloc[0] * 10000  # Normalize to initial investment

      ax2.plot(backtest_df.index, portfolio_value,
              label='Strategy Portfolio Value', color='#00d4aa', linewidth=2)
      ax2.plot(backtest_df.index, market_value,
              label='Buy & Hold Value', color='#0ea5e9', linestyle='--', linewidth=2)
      ax2.set_title(f'{ticker} Portfolio Performance', color='white', fontsize=16, fontweight='bold')
      ax2.set_xlabel('Date', color='white', fontsize=12)
      ax2.set_ylabel('Value ($)', color='white', fontsize=12)
      ax2.legend(fontsize=12)
      ax2.grid(True, alpha=0.3)
      ax2.tick_params(colors='white')
      ax2.spines['bottom'].set_color('white')
      ax2.spines['top'].set_color('white')
      ax2.spines['right'].set_color('white')
      ax2.spines['left'].set_color('white')

      plt.tight_layout()
      
      # Save plot if path is provided
      if save_path:
          plt.savefig(save_path, transparent=True, bbox_inches='tight', 
                     facecolor='none', edgecolor='none', dpi=150)
          plt.close()
      else:
          plt.show()

      # Calculate and print performance metrics
      initial_value = portfolio_value.iloc[0]
      final_value = portfolio_value.iloc[-1]
      total_return = (final_value / initial_value - 1) * 100
      
      # Fix market return calculation - should be based on the normalized market value, not raw price
      market_initial = market_value.iloc[0]  # This is 10000 (same as initial_value)
      market_final = market_value.iloc[-1]
      market_return = (market_final / market_initial - 1) * 100

      # Calculate Sharpe Ratio (assuming risk-free rate of 0.01)
      strategy_returns = portfolio_value.pct_change()
      excess_returns = strategy_returns - 0.01/252  # Daily risk-free rate
      sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

      # Calculate Maximum Drawdown
      rolling_max = portfolio_value.expanding().max()
      drawdowns = portfolio_value/rolling_max - 1.0
      max_drawdown = drawdowns.min() * 100

      print(f"\nPerformance Metrics for {ticker}:")
      print(f"Initial Portfolio Value: ${initial_value:,.2f}")
      print(f"Final Portfolio Value: ${final_value:,.2f}")
      print(f"Strategy Total Return: {total_return:.2f}%")
      print(f"Buy & Hold Return: {market_return:.2f}%")
      print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
      print(f"Maximum Drawdown: {max_drawdown:.2f}%")
      print(f"Win Rate: {(strategy_returns > 0).mean():.2%}")
      
      return {
          'total_return': total_return,
          'market_return': market_return,
          'sharpe_ratio': sharpe_ratio,
          'max_drawdown': max_drawdown,
          'win_rate': (strategy_returns > 0).mean(),
          'final_value': final_value
      }

    def predict_for_user_selection(self, selected_tickers, start_date, end_date):
        """Make predictions for user-selected stocks using the pre-trained 50-stock model"""
        print(f"Making predictions for selected stocks: {selected_tickers}")
        
        # Validate that selected stocks are in our TOP_50 list
        valid_tickers = [t for t in selected_tickers if t in self.TOP_50_SP500]
        if len(valid_tickers) != len(selected_tickers):
            invalid = set(selected_tickers) - set(valid_tickers)
            print(f"Warning: These tickers are not in TOP_50_SP500: {invalid}")
        
        # Fetch data only for selected stocks
        data = self.fetch_and_prepare_data(valid_tickers, start_date, end_date)
        # Store original close prices in a simpler format
        original_close_dict = {}
        for ticker in valid_tickers:
            ticker_data = data[data['Ticker'] == ticker].copy()
            original_close_dict[ticker] = ticker_data.set_index('Date')['Close']
        
        normalized_data = self.normalize_data(data)
        
        # Create sequences with fixed 50-stock dimensions (zeros for unselected stocks)
        X, y, dates = self.create_sequences(normalized_data, valid_tickers, original_close_dict)
        
        if len(X) == 0:
            print("No sequences created - check your data and date range")
            return None
        
        print(f"Created {len(X)} sequences with shape {X.shape}")
        
        # Make predictions
        predictions = self.model.predict(X, batch_size=32, verbose=0).flatten()
        
        # Process results for each ticker
        results = {}
        start_idx = 0
        
        for ticker in valid_tickers:
            ticker_data = data[data['Ticker'] == ticker]
            num_sequences = max(0, len(ticker_data) - self.seq_length)
            
            if num_sequences > 0:
                ticker_predictions = predictions[start_idx:start_idx + num_sequences]
                ticker_actual = y[start_idx:start_idx + num_sequences]
                
                # Inverse transform predictions to real prices
                dummy_array_pred = np.zeros((len(ticker_predictions), len(self.features)))
                dummy_array_pred[:, 0] = ticker_predictions
                ticker_predictions_real = self.scalers[ticker].inverse_transform(dummy_array_pred)[:, 0]
                
                # Actual prices are already real (from original_close parameter)
                ticker_actual_real = ticker_actual
                
                results[ticker] = {
                    'predictions': ticker_predictions_real.tolist(),
                    'dates': dates[start_idx:start_idx + num_sequences].tolist(),
                    'actual_prices': ticker_actual_real.tolist()
                }
                
                start_idx += num_sequences
        
        return results

if __name__ == "__main__":
    # Test the new 50-stock implementation
    print("=" * 60)
    print("TESTING 50-STOCK LSTM IMPLEMENTATION")
    print("=" * 60)
    
    predictor = LSTMStockPredictor('models/lstm_50_stocks_model.keras', seq_length=30)
    
    # Test 1: Train on all 50 stocks (one-time setup)
    print("\n🚀 TEST 1: Training model on all 50 S&P stocks...")
    try:
        X_test, y_test = predictor.train_model_on_all_50_stocks()
        print("✅ Training successful!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        exit(1)
    
    # Test 2: Predict for a subset of stocks (user simulation)
    print("\n🔮 TEST 2: Making predictions for user-selected stocks...")
    selected_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']  # User selects 5 stocks
    
    try:
        results = predictor.predict_for_user_selection(
            selected_stocks, 
            '2022-01-01', 
            '2024-12-31'
        )
        
        if results:
            print("✅ Predictions successful!")
            for ticker, data in results.items():
                print(f"📊 {ticker}: {len(data['predictions'])} predictions generated")
                print(f"   Latest prediction: ${data['predictions'][-1]:.2f}")
                print(f"   Actual price: ${data['actual_prices'][-1]:.2f}")
        else:
            print("❌ No predictions generated")
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    # Test 3: Backtest a single stock
    print("\n📈 TEST 3: Backtesting AAPL...")
    try:
        backtest_results = predictor.backtest_stock(
            'AAPL', 
            '2022-01-01', 
            '2024-12-31', 
            ['AAPL']  # Pass as list for consistency
        )
        
        if backtest_results is not None:
            print("✅ Backtesting successful!")
            metrics = predictor.plot_results(
                backtest_results, 
                'AAPL', 
                '2022-01-01', 
                '2024-12-31',
                save_path='static/AAPL_50stock_test.png'
            )
            print(f"📊 Strategy Return: {metrics['total_return']:.2f}%")
            print(f"📊 Buy & Hold Return: {metrics['market_return']:.2f}%")
        else:
            print("❌ Backtesting failed - insufficient data")
            
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)