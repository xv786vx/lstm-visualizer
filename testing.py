import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_financial_indicators(ticker):
  """
  Extract key financial indicators for a given company ticker using Yahoo Finance API.

  Args:
  ticker (str): Stock ticker symbol of the company (e.g. 'AAPL' for Apple Inc.).

  Returns:
  dict: A dictionary containing the indicators.
  """
  try:
    # fetch company data
    company = yf.Ticker(ticker)

    # extracting associated metrics
    financial_data = {
      'Company Name': company.info.get('shortName', 'N/A'),
      'Sector': company.info.get('sector', 'N/A'),
      'Industry': company.info.get('industry', 'N/A'),
      'Market Cap': company.info.get('marketCap', 'N/A'),
      'EPS (TTM)': company.info.get('trailingEps', 'N/A'),
      'P/E Ratio': company.info.get('trailingPE', 'N/A'),
      'PEG Ratio': company.info.get('pegRatio', 'N/A'),
      'Price-to-Book Ratio': company.info.get('priceToBook', 'N/A'),
      'Profit Margin': company.info.get('profitMargins', 'N/A'),
      'Dividend Yield': company.info.get('dividendYield', 'N/A'),
      '52-Week High': company.info.get('fiftyTwoWeekHigh', 'N/A'),
      '52-Week Low': company.info.get('fiftyTwoWeekLow', 'N/A'),
      'Beta': company.info.get('beta', 'N/A'),
    }
    return financial_data
  except Exception as e:
    print(f"Error fetching data for {ticker}: {e}")
    return None
  
print(get_financial_indicators('AAPL'))
