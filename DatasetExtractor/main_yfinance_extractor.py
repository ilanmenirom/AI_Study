import yfinance as yf
import pandas as pd

start = "2020-01-01"
end="2025-05-01"
all_symbols = ["TSLA", "GC=F","BTC-USD"]
for symbol in all_symbols:
    df = yf.download(symbol,start=start,end=end)
    #TODO: the format has a problem with placing "ticker" and date. Also need to insert a coloum of target which will represet r_i,p
    df.to_csv(r"Dataset/" + symbol + ".csv")

