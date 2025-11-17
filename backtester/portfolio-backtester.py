import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_data():
    """
    Fetch data for all portfolio components
    - Equities: S&P 500, NASDAQ
    - Bonds: 10-Year Treasury
    - Gold: GLD ETF (physical gold proxy)
    - Commodities: DBC (diversified commodity index)
    - Vol/Crisis indicators: VIX, Treasury yields
    """
    tickers = {
        # Equities
        '^IXIC': 'NASDAQ',
        '^GSPC': 'SP500',
        '^SPX': 'SP500_V2',
        
        # Fixed Income
        'TLT': 'BOND_ETF',          # Long-term treasury ETF
        'IEF': 'BOND_ETF_7-10',
        'SHY': "BOND_ETF_1-3",


        
        # Gold (changed from futures to ETF for consistency)
        'GLD': 'GOLD',              # Physical gold ETF
        
        # Commodities 
        'DBC': 'COMMODITIES',       # Invesco DB Commodity Index
        'DBE': 'COMMODITIES_ENERGY',
        'DBA': 'COMMODITIES_AGRIC',
        'DBB': 'COMMODITIES_BASE_METALS',

        
        # Risk indicators
        '^VIX': 'VIX',
    }
    
    data = {}
    for ticker, name in tickers.items():
        try:
            df = yf.download(ticker, period='max', progress=False)
            print(df)
            if df is None or df.empty:
                print(f"Skipping {ticker}: No data returned")
                continue
            
            if 'Close' in df.columns:
                data[name] = df['Close'].squeeze()
            else:
                print(f"Skipping {ticker}: No Close column found")
        except Exception as e:
            print(f"Skipping {ticker}: {str(e)}")
    
    if not data:
        raise ValueError("No data downloaded. Check internet connection.")
    
    df = pd.DataFrame(data).dropna()
    
    return df

def sma(prices, period):
    """Simple Moving Average"""
    return prices.rolling(window=period).mean()

def atr(prices, period=20):
    """Average True Range"""
    return prices.rolling(period).std()

# ============================================================================
# STRATEGY 1: EQUITIES (24%)
# ============================================================================
def equity_strategy(data, weight=0.24):
    """
    Hedged equity strategy:
    - Long S&P 500 (70%) + NASDAQ (30%)
    - Hedge 50% when SPX < 200-day MA
    """
    sp500 = data['SP500_V2']
    nasdaq = data['NASDAQ']
    
    # Trend filter: reduce exposure when below 200-day MA
    ma200 = sma(sp500, 200)
    hedge_signal = (sp500 < ma200).astype(float) * 0.5  # Reduce by 50%
    
    # Combined equity return
    sp_ret = sp500.pct_change() * 0.7
    nq_ret = nasdaq.pct_change() * 0.3
    combined = (sp_ret + nq_ret) * (1 - hedge_signal)
    
    return combined * weight

# ============================================================================
# STRATEGY 2: GOLD (19%) - SIMPLIFIED TO 100% GOLD
# ============================================================================
def gold_strategy(data, weight=0.19):
    """
    Passive gold allocation:
    - 100% physical gold ETF (GLD)
    - Buy and hold, no active signals
    """
    gold_ret = data['GOLD'].pct_change()
    return gold_ret * weight

# ============================================================================
# STRATEGY 3: FIXED INCOME (18%)
# ============================================================================
def fixed_income_strategy(data, weight=0.18):
    """
    Long-term Treasury bonds
    Uses TLT 
    """
    bond_ret = data['BOND_ETF'].pct_change()
    
    return bond_ret * weight

# ============================================================================
# STRATEGY 4: COMMODITIES (15%) - FIXED!
# ============================================================================
def commodity_strategy(data, weight=0.15):
    """
    Active trend-following on commodities:
    - Entry: Price breaks 20-day high/low in direction of 200-day MA
    - Exit: MA50 cross or 4×ATR profit target
    - Position sizing: Risk-based using ATR
    
    This is SEPARATE from gold - tracks energy, metals, agriculture
    """
    commodity = data['COMMODITIES']
    
    ma50 = sma(commodity, 50)
    ma200 = sma(commodity, 200)
    high_20 = commodity.rolling(20).max()
    low_20 = commodity.rolling(20).min()
    
    # Initialize positions
    position = np.zeros(len(commodity))
    
    # Trend-following logic
    for i in range(200, len(commodity)):
        if position[i-1] == 0:  # No position
            # Long entry: above MA200 AND breaks 20-day high
            if commodity.iloc[i] > ma200.iloc[i] and commodity.iloc[i] >= high_20.iloc[i-1]:
                position[i] = 1.0
            # Short entry: below MA200 AND breaks 20-day low
            elif commodity.iloc[i] < ma200.iloc[i] and commodity.iloc[i] <= low_20.iloc[i-1]:
                position[i] = -1.0
        else:  # Have position
            position[i] = position[i-1]
            # Exit long on MA50 cross down
            if position[i-1] == 1 and commodity.iloc[i] < ma50.iloc[i]:
                position[i] = 0
            # Exit short on MA50 cross up
            elif position[i-1] == -1 and commodity.iloc[i] > ma50.iloc[i]:
                position[i] = 0
    
    position_series = pd.Series(position, index=commodity.index)
    returns = position_series.shift(1) * commodity.pct_change()
    
    return returns * weight

# ============================================================================
# STRATEGY 5: LONG VOLATILITY (17%)
# ============================================================================
def long_vol_strategy(data, weight=0.17):
    """
    Crisis alpha strategy:
    - Buy equities after strong 63-day rallies (momentum continues)
    - Buy bonds after sharp 63-day drops (flight to safety)
    - Vol-targeted position sizing
    - Hold for 30 days, then 5-day cooldown
    """
    sp500 = data['SP500']
    bond = data['BOND_ETF'] 
    
    # 63-day return
    r_63 = sp500 / sp500.shift(63) - 1
    
    position_eq = np.zeros(len(sp500))
    position_bond = np.zeros(len(sp500))
    
    hold_days = 0
    cooldown = 0
    
    for i in range(63, len(sp500)):
        if cooldown > 0:
            cooldown -= 1
            continue
        
        if hold_days > 0:
            hold_days -= 1
            position_eq[i] = position_eq[i-1]
            position_bond[i] = position_bond[i-1]
            if hold_days == 0:
                position_eq[i] = 0
                position_bond[i] = 0
                cooldown = 5
        # Strong rally: ride momentum
        elif r_63.iloc[i] >= 0.05:
            position_eq[i] = 1.0
            hold_days = 30
        # Sharp drop: buy bonds (amplified position)
        elif r_63.iloc[i] <= -0.05:
            position_bond[i] = 1.5  # Bonds rally harder in crashes
            hold_days = 30
    
    position_eq_series = pd.Series(position_eq, index=sp500.index)
    position_bond_series = pd.Series(position_bond, index=sp500.index)
    
    # Volatility targeting
    atr_val = sp500.rolling(20).std() * np.sqrt(252)
    vol_target = (0.10 / atr_val).clip(upper=1.7)
    
    returns = (position_eq_series.shift(1) * sp500.pct_change() * vol_target.shift(1) +
               position_bond_series.shift(1) * bond.pct_change())
    
    return returns * weight

# ============================================================================
# STRATEGY 6: QIS (7%)
# ============================================================================
def qis_strategy(data, weight=0.07):
    """
    Quantitative Investment Strategies:
    - Simple 12-month momentum on equities and bonds
    - Market-neutral approach
    """
    sp500 = data['SP500']
    bond = data['BOND_ETF'] 
    
    # 12-month momentum signals
    mom_eq = (sp500 / sp500.shift(252) - 1) > 0
    mom_bond = (bond / bond.shift(252) - 1) > 0
    
    # Apply momentum filters
    eq_ret = sp500.pct_change() * mom_eq.shift(1).astype(float)
    bond_ret = bond.pct_change() * mom_bond.shift(1).astype(float)
    
    return (eq_ret + bond_ret) / 2 * weight

# ============================================================================
# BENCHMARKS
# ============================================================================
def benchmark_60_40(data):
    """Traditional 60/40 portfolio"""
    bond = data['BOND_ETF'] 
    return 0.6 * data['SP500'].pct_change() + 0.4 * bond.pct_change()

def benchmark_spy(data):
    """S&P 500 only"""
    return data['SP500'].pct_change()

def passive_portfolio(data):
    """
    Passive version of your strategy (no active signals)
    Shows the value-add of your active management
    """
    eq_ret = 0.24 * data['SP500'].pct_change()
    gold_ret = 0.19 * data['GOLD'].pct_change()
    
    bond = data['BOND_ETF'] 
    bond_ret = 0.18 * bond.pct_change()
    
    comm_ret = 0.15 * data['COMMODITIES'].pct_change()
    
    # leave at 83% invested, 17% cash with annual 2% returns
    vix_ret = 0.17 * .02/252  # Can't hold VIX so converted into cash
    
    qis_ret = 0.07 * (data['SP500'].pct_change() + bond.pct_change()) / 2
    
    return eq_ret + gold_ret + bond_ret + comm_ret + vix_ret + qis_ret

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
def calculate_metrics(returns, rf=0.03):
    """Calculate comprehensive performance metrics"""
    returns = returns.dropna()
    
    # Total return and CAGR
    total_ret = (1 + returns).prod() - 1
    years = len(returns) / 252
    cagr = (1 + total_ret) ** (1/years) - 1
    
    # Volatility
    vol = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = np.sqrt(252) * (returns.mean() - rf/252) / returns.std()
    
    # Sortino Ratio
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (cagr - rf) / downside if downside > 0 else np.nan
    
    # Max Drawdown
    cum = (1 + returns).cumprod()
    dd = (cum / cum.expanding().max() - 1).min()
    
    # Calmar Ratio
    calmar = cagr / abs(dd) if dd != 0 else np.nan
    
    return {
        'CAGR': cagr,
        'Vol': vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max DD': dd
    }

# ============================================================================
# MAIN BACKTEST
# ============================================================================
def run_backtest():
    print("="*80)
    print(" "*20 + "PORTFOLIO BACKTEST v2.0")
    print("="*80)
    print("\nDownloading data...\n")
    
    data = fetch_data()
    
    print(f"✓ Backtest Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"✓ Total Years: {len(data)/252:.1f}")
    print(f"✓ Available Assets: {', '.join(data.columns)}\n")
    
    # Construct active strategy
    active = (equity_strategy(data) + 
              gold_strategy(data) + 
              fixed_income_strategy(data) + 
              commodity_strategy(data) +
              long_vol_strategy(data) + 
              qis_strategy(data))
    
    strategies = {
        'Active Strategy': active,
        '60/40 Benchmark': benchmark_60_40(data),
        'S&P 500 Only': benchmark_spy(data),
        'Passive (No Signals)': passive_portfolio(data)
    }
    
    # Calculate metrics
    results = pd.DataFrame({name: calculate_metrics(ret) 
                           for name, ret in strategies.items()}).T
    
    # Format results
    for col in ['CAGR', 'Vol', 'Max DD']:
        results[col] = results[col].apply(lambda x: f"{x:.2%}")
    for col in ['Sharpe', 'Sortino', 'Calmar']:
        results[col] = results[col].apply(lambda x: f"{x:.2f}")
    
    print("="*80)
    print(" "*25 + "PERFORMANCE METRICS")
    print("="*80)
    print(results.to_string())
    print("\n")
    
    # Create visualizations
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Cumulative returns
    for name, ret in strategies.items():
        cumulative = (1 + ret).cumprod()
        axes[0].plot(cumulative.index, cumulative.values, label=name, linewidth=2)
    axes[0].set_title('Cumulative Returns (Growth of $1)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale('log')
    
    # 2. Drawdowns
    for name, ret in strategies.items():
        cum = (1 + ret).cumprod()
        dd = (cum / cum.expanding().max() - 1) * 100
        axes[1].plot(dd.index, dd.values, label=name, linewidth=2)
    axes[1].set_title('Drawdowns Over Time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].legend(loc='lower left')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Rolling 3-year Sharpe
    for name, ret in strategies.items():
        rolling_sharpe = ret.rolling(252*3).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0,
            raw=True
        )
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=2)
    axes[2].set_title('Rolling 3-Year Sharpe Ratio', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Sharpe Ratio')
    axes[2].legend(loc='upper left')
    axes[2].grid(alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
    print("✓ Charts saved: backtest_results.png\n")
    
    # Crisis performance
    crises = {
        '2000 Dot-com': ('2000-03-01', '2002-10-01'),
        '2008 Financial': ('2007-10-01', '2009-03-01'),
        '2020 COVID': ('2020-02-01', '2020-04-01'),
        '2022 Inflation': ('2022-01-01', '2022-10-01'),
    }
    
    crisis_perf = {}
    for crisis_name, (start, end) in crises.items():
        try:
            period_returns = {
                name: (1 + ret[start:end]).prod() - 1
                for name, ret in strategies.items()
            }
            crisis_perf[crisis_name] = period_returns
        except:
            pass
    
    if crisis_perf:
        crisis_df = pd.DataFrame(crisis_perf).T
        crisis_df = crisis_df.map(lambda x: f"{x:.1%}")
        print("="*80)
        print(" "*25 + "CRISIS PERFORMANCE")
        print("="*80)
        print(crisis_df.to_string())
        print("\n")
    
    return data, strategies, results

if __name__ == "__main__":
    data, strategies, results = run_backtest()