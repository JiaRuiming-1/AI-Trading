## Project Instruction

## Get Date
We load data from [Tushare](https://tushare.pro/) platform, which is a quant trading data supplier and most of data can be use for free.
There is an example of coding in `Tushare_Coding.ipynb` file you can get view.

install stockstats
close_stats = stockstats.StockDataFrame.retype(close) # column: date row: ticker_name

·## one day diff value
close_stats[['aapl','aapl_delta','aapl_-1_d']]

·## macd
close_stats['close'] = close_stats['aapl']
close_stats[['macd','macds','macdh']].plot(figsize=(10,5), grid=True)

·## kdj
import random
close_stats['high'] = close_stats['aapl'].apply(lambda x: max(x,x*random.random()))
close_stats[['kdjk','kdjd','kdjj']].plot(figsize=(20,5), grid=True)

`## document https://pypi.org/project/stockstats/

`## sma
close_stats[['close','close_10_sma','close_60_sma','close_300_sma']].plot(figsize=(20,10), grid=True)

## review math concept
