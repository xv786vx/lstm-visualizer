import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

maybe = yf.download("MSFT", start='2020-01-01', end='2021-01-01')
print(maybe)
