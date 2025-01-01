import yfinance as yf
print(yf.download('AAPL', period='max', interval='1d'))