import yfinance as yf
import pandas as pd

start = "2024-01-01"
end="2025-05-01"
all_symbols = ["TSLA", "GC=F","BTC-USD"]
for symbol in all_symbols:
    df = yf.download(symbol,start=start,end=end)
    df.to_csv(r"../Dataset/" + symbol + ".csv")
