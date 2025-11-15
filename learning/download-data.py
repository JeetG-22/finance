import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

#download SPY data from 2010-2024
spy = yf.download("SPY", start="2010-01-01", end="2024-12-31")
print(type(spy))
print(spy.columns)
print(spy)

spy["Close"].plot(figsize=(12,6), title="SPY Price History")
plt.ylabel("Price($)")
plt.show()