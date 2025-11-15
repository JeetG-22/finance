import yfinance as yf
import pandas as pd 
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(
    filename='stock_updates.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def update_stock_metrics():
    try:
        tickers = ["FXAIX", "QQQ", "VZ", "T", "UNH", "DKNG", "LULU"]
        data = []
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            data.append({
                
                # Basic Info From Summary Page
                'Ticker': ticker,
                'Price': info.get('currentPrice'),
                'Market Cap': info.get('marketCap'),
                'P/E Ratio': info.get('trailingPE'),
                # Risk
                'Beta': info.get('beta'),
                # Earnings
                'EPS': info.get('trailingEps'),
                # Dividend
                'Dividend Yield': info.get('dividendYield'),
                'Payout Ratio': info.get('payoutRatio'),
                # 52-Week Range
                '52W Low': info.get('fiftyTwoWeekLow'),
                '52W High': info.get('fiftyTwoWeekHigh'),
                
                # Valuation Ratios
                'Forward P/E': info.get('forwardPE'),
                'PEG Ratio': info.get('pegRatio'),
                'Price/Book': info.get('priceToBook'),
                'Price/Sales': info.get('priceToSalesTrailing12Months'),
                'EV/Revenue': info.get('enterpriseToRevenue'),
                'EV/EBITDA': info.get('enterpriseToEbitda'),
                
                # Profitability
                'Profit Margin': info.get('profitMargins'),
                'Operating Margin': info.get('operatingMargins'),
                'ROE': info.get('returnOnEquity'),
                'ROA': info.get('returnOnAssets'),
                
                # Growth
                'Revenue Growth': info.get('revenueGrowth'),
                'Earnings Growth': info.get('earningsGrowth'),
                
                # Timestamp
                'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        df = pd.DataFrame(data)
        # Expand the ~ to full path
        file_path = os.path.expanduser("~/Documents/Personal Portfolio.xlsx")
        
        # Check if file exists
        if os.path.exists(file_path):
            # File exists - replace the Evaluation sheet
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name="Evaluation", index=False)
        else:
            # File doesn't exist - create it
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, sheet_name="Evaluation", index=False)
                
        #Update logging info
        logging.info(f"Successfully updated {len(tickers)} stocks")
        print(f"Updated at {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        logging.error(f"Error updating stocks: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    update_stock_metrics()