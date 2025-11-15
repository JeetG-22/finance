import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

spy = yf.download("SPY", start="2010-01-01", end="2024-12-31")

spy["MA50"] = spy["Close"].rolling(window=50).mean()
spy["MA200"] = spy["Close"].rolling(window=200).mean()

print(spy[["Close", "MA50", "MA200"]].tail(10))

# Plot everything
plt.figure(figsize=(14, 7))
plt.plot(spy['Close'], label='SPY Price', linewidth=1)
plt.plot(spy['MA50'], label='50-day MA', linewidth=1.5)
plt.plot(spy['MA200'], label='200-day MA', linewidth=2)
plt.title('SPY with Moving Averages')
plt.legend()
plt.ylabel('Price ($)')
plt.show()
