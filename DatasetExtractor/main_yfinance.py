import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def download_stock_data(symbols, start_date, end_date):
    """
    Download stock data for multiple symbols and format it properly
    """
    all_data = {}
    
    # Download each symbol separately to avoid MultiIndex
    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date)
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Rename columns to include symbol
        df.columns = ['date'] + [f'{col[0].lower()}' for col in df.columns[1:]]
        
        # Calculate returns relative to previous day's close
        # Shift close price up by 1 to get previous day's close
        prev_close = df['close'].shift(1)
        
        # Normalize price columns by previous day's close
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = df[col] / prev_close
            
        # For volume, we'll use percentage change instead of direct division
        df['volume'] = df['volume'].pct_change()
        
        # Drop the first row since it has NaN values after normalization
        df = df.dropna()
        
        # Add target column (next day's normalized close price)
        df['target'] = df['close'].shift(-1)
        df['day'] = df['date'].dt.dayofweek
        # Drop the last row since it has NaN target
        df = df.dropna()
        
        # Save to CSV
        df.to_csv(f'Dataset/{symbol}.csv', index=False)
        
        print(f"Downloaded and saved data for {symbol}")

if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL', 'GOOGL', 'MSFT' ,"GC=F","BTC-USD"]  # Add your symbols here
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Last year of data
    
    download_stock_data(symbols, start_date, end_date) 