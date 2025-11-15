import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

spy = yf.download("SPY", start="2010-01-01", end="2024-12-31")

spy["MA200"] = spy["Close"].rolling(window=200).mean()

#creating signals
"""
1 = bullish (price > MA200)
0 = bearest (price < MA200)
"""
spy["Signal"] = 0
spy.loc[spy["Close"].squeeze() > spy["MA200"].squeeze(), "Signal"] = 1

#find when signal changes (crossovers)
"""
1 = crossed above MA200 (buy signal)
2 = crossed below MA200 (sell signal)
0 = no change
"""
spy["Position_Change"] = spy["Signal"].diff()

print(spy.tail(30)) 

# Look at crossovers
crossovers = spy[spy['Position_Change'] != 0].copy()
print("Buy/Sell Signals:")
print(crossovers[['Close', 'MA200', 'Signal', 'Position_Change']].head(20))

# Plot with buy/sell markers
plt.figure(figsize=(14, 7))
plt.plot(spy['Close'], label='SPY Price', alpha=0.7)
plt.plot(spy['MA200'], label='200-day MA', linewidth=2)

# Mark buy signals (green triangles)
buys = spy[spy['Position_Change'] == 1.0]
plt.scatter(buys.index, buys['Close'], color='green', marker='^', 
            s=100, label='BUY', zorder=5)

# Mark sell signals (red triangles)
sells = spy[spy['Position_Change'] == -1.0]
plt.scatter(sells.index, sells['Close'], color='red', marker='v', 
            s=100, label='SELL', zorder=5)

plt.title('SPY Trading Signals')
plt.legend()
plt.ylabel('Price ($)')
plt.show()
