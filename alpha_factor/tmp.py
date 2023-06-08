tmp = universe.loc[(universe.index>='2023-05-01 00:00:00') & (universe.index<='2023-05-20 00:00:00') & (universe.ts_code=='DOTUSDT')]

#tmp['alpha_kama'] = tmp['alpha_kama'] + 0.3
#tmp['alpha_ppo'] = tmp['alpha_ppo']/30 + 0.3
#tmp['alpha_t1'] = tmp['alpha_t1']/100 + .3
#tmp['alpha_t2'] = tmp['alpha_t2']/50 + .33
#tmp['alpha_t3'] = tmp['alpha_t3'] + .33
#tmp['alpha_t4'] = tmp['alpha_t4']/30 + 0.33
tmp['alpha_t5'] = tmp['alpha_t5'] + 0.3
#tmp['alpha_t6'] = tmp['alpha_t6']/200 + 0.3
#tmp['alpha_t8'] = tmp['alpha_t8']/200 + 0.3
#tmp['alpha_t9'] = tmp['alpha_t9']/200 + 0.3
#tmp['alpha_t10'] = tmp['alpha_t10'] + 0.3
#tmp['alpha_019'] = tmp['alpha_019']/40 + 0.3
tmp['alpha_075'] = tmp['alpha_075'] + 0.3

tmp['close'] = tmp['close'].shift(-2)
tmp[['close', 'alpha_075', 'alpha_t5']].plot(grid=True)


def rolling_jump(sr, win, interval):
    
    if win>=interval:
        result = [np.nan]*(win-interval)
        for i in range(win, len(sr), interval):
            result.extend([np.nan]*(interval-1) + [sr[i-win:i].mean()])
    else:
        result = []
        for i in range(interval, len(sr), interval):
            result.extend([np.nan]*(interval-1) + [sr[i-win:i].mean()])

    print(i, len(result))
    if len(sr) > len(result):
        result.extend([np.nan]*(len(sr) - len(result)))
    return pd.Series(result, sr.index)


def alpha_t1(df):
    def cal_(data):
        close_4h = rolling_jump(data['close'], 1, 4).fillna(method='ffill')
        data['alpha_t1'] = -(close_4h - data['close'].shift(4))\
                            /data['close'].rolling(20).std()
        data['alpha_t1'] = rolling_jump(data['alpha_t1'], 3, 4).fillna(method='ffill')
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t1(universe) 


def alpha_t2(df):
    def cal_(data):
        data['buy_percent'] = data['buy_amount']/(data['amount'] - data['buy_amount'])
        data['alpha_t2'] = rolling_jump(-data['buy_percent'], 4, 4).fillna(method='ffill')
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df
universe = alpha_t2(universe) 


def alpha_t3(df):
    def cal_(df): # 040
        close_4h = rolling_jump(df['close'], 1, 4).fillna(method='ffill')
        cond1 = (close_4h > close_4h.shift(1)) & (df['close'] > df['close'].shift(1))
        cond2 = (close_4h < close_4h.shift(1)) & (df['close'] < df['close'].shift(1))
        df['section1'] = np.where(cond1, df['close'].diff(1), 0)
        df['section2'] = np.where(cond2, df['close'].diff(1), 0)
        df['section1'] = df['section1'].rolling(14).sum()
        df['section2'] = df['section2'].rolling(14).sum()
        return df

    df = my_groupby(df, 'ts_code', cal_)
    df['alpha_t3'] = df['section2'] - df['section1']
    df = df.drop(columns=['section1', 'section2'])
    return df

universe = alpha_t3(universe)


def alpha_t4(df):
    def cal_(data):
        close_4h = rolling_jump(data['close'], 1, 4).fillna(method='ffill')
        low_4h = rolling_jump(data['low'].rolling(4).min(), 1, 4).fillna(method='ffill')
        wave = data['high'].rolling(14).max() - data['low'].rolling(14).min()
        data['alpha_t4'] = (close_4h - low_4h)/wave
        data['alpha_t4'] = rolling_jump(-data['alpha_t4'], 2, 4).fillna(method='ffill')
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t4(universe)


def alpha_t5(df):
    def cal_(data):
        high_4h = rolling_jump(data['high'], 1, 4).fillna(method='ffill')
        low_4h = rolling_jump(data['low'].rolling(4).min(), 1, 4).fillna(method='ffill')
        data['alpha_t5'] = -np.sign(data['log-ret']) * (high_4h - low_4h)/ ts_rank(data['volume'].rolling(2).mean(), 14)
        data['alpha_t5'] = rolling_jump(data['alpha_t5'], 2, 3).fillna(method='ffill')
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t5(universe)


def alpha_t6(df):
    def cal_(data):
        cond = data['close'].diff(4)/data['close'].rolling(14).std()
        data['alpha_t6'] = rolling_jump(cond.diff(5), 4, 4).fillna(method='ffill')
        data['alpha_t6'] = data['alpha_t6'].rolling(7).mean()
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t6(universe)


def alpha_t8(df):
    def cal_(data):
        sectionM = data['close'].rolling(60).mean().fillna(1.)
        sectionX = data['close'].rolling(26).mean().fillna(1.)
        sectionY = data['close'].rolling(7).mean().fillna(1.)
        data['alpha_t8'] = sectionY.rolling(7).apply(
                            lambda y: np.polyfit(sectionX.loc[y.index], y, deg=1)[0])
        
        es_1 = sectionY/sectionX
        es_2 = sectionX/sectionM
        data['alpha_t8'] = data['alpha_t8'] / (ts_rank(data['volume'].rolling(2).mean(), 14) * (0.6 * es_1 + 0.4 * es_2))
        data['alpha_t8'] = rolling_jump(data['alpha_t8'], 2, 6).fillna(method='ffill')
        return data

    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t8(universe)


def alpha_t9(df):
    def cal_(df):
        benchmark_close = index_df.loc[df.index]['close'].rolling(4).mean()
        close_4h = rolling_jump(df['close'], 2, 4).fillna(method='ffill')
        df['alpha_t9'] = close_4h.rolling(14).apply(
                            lambda y: np.polyfit(benchmark_close.loc[y.index], y, deg=1)[0])
        df['alpha_t9'] = df['alpha_t9'].diff(2)
        df['alpha_t9'] = rolling_jump(df['alpha_t9'], 1, 3).fillna(method='ffill')
        return df
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t9(universe)


def alpha_t10(df):
    def cal1_(data, win_len=5):
        section = data['close']
        section = -np.where(section < section.shift(win_len),
                                   (section - section.shift(win_len)) / section.shift(win_len),
                                   np.where(section > section.shift(win_len),
                                            (section - section.shift(win_len)) / section, 0.))
        return pd.Series(section, index=data.index)
    
    def cal2_(data):
        section1 = cal1_(data, 4)
        section2 = rolling_jump(section1, 2, 4).fillna(method='ffill')
        #others = section2.rolling(5).mean()
        data['alpha_t10'] = np.where((section1>0)&(section2>0), -section2,
                                    np.where((section1<0)&(section2<0), -section2, 0))
        data['alpha_t10'] = rolling_jump(data['alpha_t10'], 2, 4).fillna(method='ffill')
        
        return data
    
    df = my_groupby(df, 'ts_code', cal2_)
    return df

universe = alpha_t10(universe)


def alpha019(df):
    def cal_(df):
        df['alpha_019'] = -np.where(df['close'] < df['close'].shift(5),
                                   (df['close'] - df['close'].shift(5)) / df['close'].shift(5),
                                   np.where(df['close'] > df['close'].shift(5),
                                            (df['close'] - df['close'].shift(5)) / df['close'], 0.))
        
        df['alpha_019'] = ts_rank(df['alpha_019'], 20)/20
        df['alpha_019'] = rolling_jump(df['alpha_019'], 3, 4).fillna(method='ffill')
        return df

    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha019(universe)
