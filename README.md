# LSTM Stock Predictor

A multi-stock LSTM-based stock prediction and backtesting application with portfolio management capabilities.

## Features

- **Multi-Stock Support**: Train a single LSTM model on up to 8 stocks simultaneously
- **Advanced Technical Indicators**: RSI, Moving Averages, VWAP, EMA, Volatility
- **Portfolio Management**: Buy/sell signals with position management
- **Performance Metrics**: Total return, Sharpe ratio, maximum drawdown, win rate
- **Interactive UI**: Select stocks from popular tickers, adjust sequence length
- **Real-time Training**: Server-sent events for live progress updates

## Architecture

- **Backend**: Flask API with LSTM model training and prediction
- **Frontend**: React with Tailwind CSS for modern UI
- **Model**: Multi-stock LSTM with dropout regularization
- **Data**: Yahoo Finance API for real-time stock data

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm start
```

## Usage

1. Select up to 8 stocks from the provided list
2. Choose sequence length (20-40 days)
3. Set date range (default: 2020-01-01 to 2025-01-01)
4. Click "Generate Predictions"
5. View performance metrics and prediction plots

## API Endpoints

- `POST /train` - Submit training request
- `GET /train/stream` - Stream training progress and results
- `POST /validate-tickers` - Validate stock tickers
- `GET /earliest-start-date` - Get earliest available date for a ticker

## Model Details

The LSTM model uses:

- Sequence length: 20-40 days (user configurable)
- Features: Close price, daily returns, 50MA, 200MA, volatility, RSI, VWAP, 20EMA
- Architecture: LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)
- Training: Adam optimizer, MSE loss, 20 epochs

## Performance Metrics

- **Total Return**: Strategy performance vs buy-and-hold
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Final Portfolio Value**: Total value after trading period
