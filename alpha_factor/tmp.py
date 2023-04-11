def alpha149(df):
    ####REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)
    benchmark_close = df.groupby('date').apply(benchmark_close_day)
    benchmark_return = benchmark_return_day(benchmark_close).cumsum()
    def cal_(df):
        cond = benchmark_return.loc[df.index].copy()
        df['section1'] = df['log-ret'].cumsum()
        df['section1'] = np.where(cond<0, df['section1'], 0)
        df['section2'] = np.where(cond<0, cond, 0)
        df['alpha_149'] = df['section1'].rolling(60).apply(lambda y: np.polyfit(df.loc[y.index]['section2'], y, deg=1)[0])
        return df
    
    df = df.groupby('ts_code').apply(cal_)
    return df
