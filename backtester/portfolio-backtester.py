# backtest_framework.py

import pandas as pd
import numpy as np
import yfinance as yf
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class BacktestConfig:
    """Configuration for the backtest"""
    start_date: str = '2010-01-01'
    end_date: str = '2024-12-31'
    initial_capital: float = 1_000_000
    rebalance_frequency: str = 'M'  # M=monthly, Q=quarterly
    
@dataclass
class PortfolioWeights:
    """Target allocation percentages"""
    equities: float = 0.24
    fixed_income: float = 0.18
    gold_silver: float = 0.19
    commodities: float = 0.15
    long_vol: float = 0.17
    qis: float = 0.07
    
    def validate(self):
        total = (self.equities + self.fixed_income + self.gold_silver + 
                self.commodities + self.long_vol + self.qis)
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1.0"


class Strategy(ABC):
    """Base class for all strategies"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data = {}
        
    @abstractmethod
    def download_data(self):
        """Download required price data"""
        pass
    
    @abstractmethod
    def generate_signals(self):
        """Generate buy/sell signals"""
        pass
    
    @abstractmethod
    def calculate_returns(self) -> pd.Series:
        """Return daily returns series"""
        pass


class EquityStrategy(Strategy):
    """24% Equities with MA-based inverse ETF hedging"""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.tickers = ['SPY', 'QQQ', 'SH', 'PSQ']  # SH=inverse SPY, PSQ=inverse QQQ
        
    def download_data(self):
        print("Downloading equity data...")
        self.data = yf.download(self.tickers, 
                               start=self.config.start_date,
                               end=self.config.end_date)['Close']
        
    def generate_signals(self):
        """
        Rules from your partner's doc:
        - For SPY: if MA50 > MA200 → long SPY, else short via SH
        - For QQQ: if MA50 > MA200 → long QQQ, else short via PSQ
        """
        self.signals = pd.DataFrame(index=self.data.index)
        
        # SPY signals
        spy_ma50 = self.data['SPY'].rolling(50).mean()
        spy_ma200 = self.data['SPY'].rolling(200).mean()
        self.signals['SPY_long'] = (spy_ma50 > spy_ma200).astype(int)
        self.signals['SH_long'] = (~(spy_ma50 > spy_ma200)).astype(int)
        
        # QQQ signals  
        qqq_ma50 = self.data['QQQ'].rolling(50).mean()
        qqq_ma200 = self.data['QQQ'].rolling(200).mean()
        self.signals['QQQ_long'] = (qqq_ma50 > qqq_ma200).astype(int)
        self.signals['PSQ_long'] = (~(qqq_ma50 > qqq_ma200)).astype(int)
        
    def calculate_returns(self) -> pd.Series:
        """Calculate daily returns based on signals"""
        returns = self.data.pct_change()
        
        # 60% SPY, 40% QQQ allocation within equity sleeve
        spy_ret = (self.signals['SPY_long'].shift(1) * returns['SPY'] + 
                   self.signals['SH_long'].shift(1) * returns['SH']) * 0.6
        
        qqq_ret = (self.signals['QQQ_long'].shift(1) * returns['QQQ'] + 
                   self.signals['PSQ_long'].shift(1) * returns['PSQ']) * 0.4
        
        return spy_ret + qqq_ret


class FixedIncomeStrategy(Strategy):
    """18% Fixed Income"""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.tickers = ['TLT', 'IEF']  # Long-term and intermediate treasuries
        
    def download_data(self):
        print("Downloading fixed income data...")
        self.data = yf.download(self.tickers,
                               start=self.config.start_date,
                               end=self.config.end_date)['Close']
        
    def generate_signals(self):
        # Simple buy-and-hold for bonds
        self.signals = pd.DataFrame(1, index=self.data.index, columns=self.tickers)
        
    def calculate_returns(self) -> pd.Series:
        returns = self.data.pct_change()
        # 55% long-term, 45% intermediate
        return 0.55 * returns['TLT'] + 0.45 * returns['IEF']


class GoldSilverStrategy(Strategy):
    """19% Gold & Silver (80% physical, 20% miners)"""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.tickers = ['GLD', 'SLV', 'GDX', 'GDXJ']
        
    def download_data(self):
        print("Downloading gold/silver data...")
        self.data = yf.download(self.tickers,
                               start=self.config.start_date,
                               end=self.config.end_date)['Close']
        
    def generate_signals(self):
        # Buy and hold
        self.signals = pd.DataFrame(1, index=self.data.index, columns=self.tickers)
        
    def calculate_returns(self) -> pd.Series:
        returns = self.data.pct_change()
        # 70% GLD, 10% SLV (physical), 15% GDX, 5% GDXJ (miners)
        return (0.70 * returns['GLD'] + 
                0.10 * returns['SLV'] + 
                0.15 * returns['GDX'] + 
                0.05 * returns['GDXJ'])


class CommodityStrategy(Strategy):
    """15% Commodities with Donchian breakout"""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.tickers = ['DBC', 'USO', 'UNG']  # Broad commodities, oil, nat gas
        
    def download_data(self):
        print("Downloading commodity data...")
        self.data = yf.download(self.tickers,
                               start=self.config.start_date,
                               end=self.config.end_date)['Close']
        
    def generate_signals(self):
        """
        Donchian breakout from your partner's doc:
        - Buy when price > 20-day high
        - Sell when price < 20-day low
        """
        self.signals = pd.DataFrame(index=self.data.index)
        
        for ticker in self.tickers:
            high_20 = self.data[ticker].rolling(20).max()
            low_20 = self.data[ticker].rolling(20).min()
            ma200 = self.data[ticker].rolling(200).mean()
            
            # Only long if above MA200 (trend filter)
            long_signal = ((self.data[ticker] >= high_20) & 
                          (self.data[ticker] > ma200))
            
            # Short if below 20-day low AND below MA200
            short_signal = ((self.data[ticker] <= low_20) & 
                           (self.data[ticker] < ma200))
            
            # Convert to position: 1 = long, -1 = short, 0 = flat
            position = pd.Series(0, index=self.data.index)
            position[long_signal] = 1
            position[short_signal] = -1
            position = position.replace(0, method='ffill')  # Hold position
            
            self.signals[ticker] = position
        
    def calculate_returns(self) -> pd.Series:
        returns = self.data.pct_change()
        # Equal weight across commodities
        strategy_ret = sum(self.signals[ticker].shift(1) * returns[ticker] / len(self.tickers)
                          for ticker in self.tickers)
        return strategy_ret


class LongVolStrategy(Strategy):
    """17% Long Volatility - simplified with leveraged ETFs"""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.tickers = ['SPY', 'UPRO', 'SPXU', 'QQQ', 'TQQQ', 'SQQQ']
        self.lookback = 63
        self.threshold = 0.05
        self.hold_cap = 30
        self.cooloff = 5
        
    def download_data(self):
        print("Downloading long vol data...")
        self.data = yf.download(self.tickers,
                               start=self.config.start_date,
                               end=self.config.end_date)['Close']
        
    def generate_signals(self):
        """
        From your partner's doc:
        - Calculate 63-day return
        - If >= 5%: go long (via 3x ETF)
        - If <= -5%: go short (via 3x inverse ETF)
        - Hold for max 30 days
        - 5-day cooloff after close
        """
        self.signals = pd.DataFrame(0, index=self.data.index, 
                                   columns=['SPY_position', 'QQQ_position'])
        
        # SPY-based signals
        spy_ret_63 = self.data['SPY'].pct_change(periods=self.lookback)
        position_spy = 0
        days_held = 0
        cooloff_counter = 0
        
        for i in range(self.lookback, len(self.data)):
            idx = self.data.index[i]
            
            if cooloff_counter > 0:
                cooloff_counter -= 1
                continue
                
            if position_spy != 0:
                days_held += 1
                self.signals.loc[idx, 'SPY_position'] = position_spy
                if days_held >= self.hold_cap:
                    position_spy = 0
                    days_held = 0
                    cooloff_counter = self.cooloff
            else:
                r = spy_ret_63.iloc[i]
                if pd.notna(r):
                    if r >= self.threshold:
                        position_spy = 1  # Long via UPRO
                    elif r <= -self.threshold:
                        position_spy = -1  # Short via SPXU
                self.signals.loc[idx, 'SPY_position'] = position_spy
        
        # Similar for QQQ (omitted for brevity)
        
    def calculate_returns(self) -> pd.Series:
        returns = self.data.pct_change()
        
        # When position = 1, use UPRO; when -1, use SPXU
        spy_ret = pd.Series(0, index=self.data.index)
        spy_ret[self.signals['SPY_position'] == 1] = returns['UPRO']
        spy_ret[self.signals['SPY_position'] == -1] = returns['SPXU']
        
        return spy_ret.shift(1)  # 50% SPY, 50% QQQ would go here


class QISStrategy(Strategy):
    """7% Quantitative Investment Strategies (simplified trend)"""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.tickers = ['SPY', 'EFA', 'EEM', 'TLT', 'DBC']
        
    def download_data(self):
        print("Downloading QIS data...")
        self.data = yf.download(self.tickers,
                               start=self.config.start_date,
                               end=self.config.end_date)['Close']
        
    def generate_signals(self):
        """Simple trend following across multiple assets"""
        self.signals = pd.DataFrame(index=self.data.index)
        
        for ticker in self.tickers:
            ma50 = self.data[ticker].rolling(50).mean()
            self.signals[ticker] = (self.data[ticker] > ma50).astype(int)
        
    def calculate_returns(self) -> pd.Series:
        returns = self.data.pct_change()
        # Equal weight trend following
        return sum(self.signals[ticker].shift(1) * returns[ticker] / len(self.tickers)
                  for ticker in self.tickers)


class PortfolioBacktester:
    """Main backtester that combines all strategies"""
    
    def __init__(self, config: BacktestConfig, weights: PortfolioWeights):
        self.config = config
        self.weights = weights
        self.weights.validate()
        
        # Initialize all strategies
        self.strategies = {
            'equities': EquityStrategy(config),
            'fixed_income': FixedIncomeStrategy(config),
            'gold_silver': GoldSilverStrategy(config),
            'commodities': CommodityStrategy(config),
            'long_vol': LongVolStrategy(config),
            'qis': QISStrategy(config)
        }
        
    def run(self):
        """Execute the full backtest"""
        print("="*60)
        print("RUNNING FULL PORTFOLIO BACKTEST")
        print("="*60)
        
        # Download all data
        for name, strategy in self.strategies.items():
            strategy.download_data()
            strategy.generate_signals()
        
        # Calculate weighted returns
        returns = {}
        for name, strategy in self.strategies.items():
            weight = getattr(self.weights, name)
            returns[name] = strategy.calculate_returns() * weight
            print(f"✓ {name.replace('_', ' ').title()}: {weight*100:.0f}% allocation")
        
        # Combine all returns
        self.total_returns = pd.DataFrame(returns).sum(axis=1)
        self.cumulative_returns = (1 + self.total_returns).cumprod()
        
        # Calculate statistics
        self.calculate_statistics()
        self.plot_results()
        
    def calculate_statistics(self):
        """Calculate performance metrics"""
        cum_ret = self.cumulative_returns.iloc[-1] - 1
        annual_ret = (1 + cum_ret) ** (252 / len(self.total_returns)) - 1
        volatility = self.total_returns.std() * np.sqrt(252)
        sharpe = annual_ret / volatility if volatility > 0 else 0
        
        # Max drawdown
        cummax = self.cumulative_returns.cummax()
        drawdown = (self.cumulative_returns - cummax) / cummax
        max_dd = drawdown.min()
        
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Total Return:      {cum_ret*100:.2f}%")
        print(f"Annual Return:     {annual_ret*100:.2f}%")
        print(f"Volatility:        {volatility*100:.2f}%")
        print(f"Sharpe Ratio:      {sharpe:.2f}")
        print(f"Max Drawdown:      {max_dd*100:.2f}%")
        print("="*60)
        
    def plot_results(self):
        """Create visualization"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Cumulative returns
        ax1.plot(self.cumulative_returns, label='Portfolio', linewidth=2)
        ax1.set_title('Portfolio Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return ($1 invested)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        cummax = self.cumulative_returns.cummax()
        drawdown = (self.cumulative_returns - cummax) / cummax
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ===== USAGE =====
if __name__ == "__main__":
    # Configure backtest
    config = BacktestConfig(
        start_date='2010-01-01',
        end_date='2024-12-31',
        initial_capital=1_000_000
    )
    
    # Set portfolio weights
    weights = PortfolioWeights(
        equities=0.24,
        fixed_income=0.18,
        gold_silver=0.19,
        commodities=0.15,
        long_vol=0.17,
        qis=0.07
    )
    
    # Run backtest
    backtester = PortfolioBacktester(config, weights)
    backtester.run()