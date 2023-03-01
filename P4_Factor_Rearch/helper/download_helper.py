import pandas as pd
from tqdm import tqdm
import time

def _get_daily(ts, ts_code='000001.SZ', start_date='', end_date=''):
    for _ in range(3):
        try:
            df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date,
                            adj='qfq',ma=[10, 60, 120])
            return df
        except:
           time.sleep(1)

def get_Daily_All(ts, ts_code_list, start_date, end_date):
    '''
    param: ts: Tushare object
           ts_code_list: donwload ticks list
           start_date, end_date: trade date range
    :return: DateFrame
            comlumns: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount, ma...
    '''
    stocks_daily = pd.DataFrame()
    for ts_code in tqdm(ts_code_list, desc='ticker/tickers'):
        stocks_daily = stocks_daily.append(_get_daily(ts, ts_code=ts_code, start_date=start_date, end_date=end_date))
    return stocks_daily


