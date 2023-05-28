def alpha_t1(df):
    def cal_(data):
        #wave = data['high'].rolling(4).max() - data['low'].rolling(4).min()
        data['alpha_t1'] = (data['close'].rolling(4).mean() \
                            - data['close'].rolling(14).mean())/data['close'].rolling(14).std()
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    df['alpha_t1'] = df['alpha_t1'] + 0.2 * df['rsi_6']
    return df

universe = alpha_t1(universe)

def alpha_t2(df):
    def cal_(data):
        data['alpha_t2'] = data['log-ret'].rolling(4).sum()/(data['high'] - data['low']).rolling(6).sum()
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t2(universe)

def alpha_t3(df):
    def cal_(data):
        wave = data['high'].rolling(12).max() - data['low'].rolling(12).min()
        data['alpha_t3'] = (data['close'] - data['low'].rolling(4).mean())/wave
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t3(universe)


def alpha_t4(df): #
    # -1 * sma(ts_rank(rank(self.low), 9)) * sma(rank(close),6)
    def cal1_(data):
        data['alpha_t4'] = data['low'].rolling(12).apply(lambda x: rankdata(x)[-1])
        #data['close_r'] = data['close'].rolling(120).apply(lambda x: x.rank(pct=True)[-1])
        data['alpha_t4'] = -data['alpha_t4'].rolling(4).mean() * data['close_r']
        return data
    
    df['close_r'] = df.groupby('ts_code')['close'].apply(lambda x: x.rolling(4).mean())
    df['close_r'] = df.groupby('trade_date')['close_r'].rank(pct=True)
    df = my_groupby(df, 'ts_code', cal1_)
    df = df.drop(columns=['close_r'])
    return df

universe = alpha_t4(universe)

def keep_top_bottom(df, features, bottom=0.35, top=0.65):
    def cal_(data):
        for feature in features:
            tv = data[feature].quantile(top)
            bv = data[feature].quantile(bottom)
            data[feature] = pd.Series(np.where(data[feature]>=tv, data[feature], 
                                               np.where(data[feature]<=bv, data[feature], 0)), index=data.index)
        return data
    
    all_df = pd.DataFrame()
    for dt in tqdm(df.index.unique()):
        tmp = df.loc[df.index == dt]
        all_df = all_df.append(cal_(tmp))
    return all_df.sort_values(by=['date', 'ts_code'])



## excellent negative 'alpha_022', 'alpha_019', 'alpha_rsi', 'alpha_cci', 'alpha_srsi', 'alpha_wt', 
##                    'alpha_t1', 'alpha_t2', 'alpha_t3', 'alpha_021', 'alpha_022', 'alpha_078', 

## excellent positive 'alpha_019', 'alpha_018', 'alpha_111', 'alpha_040',

nagative_field = ['alpha_rsi', 'alpha_cci', 'alpha_srsi', 'alpha_wt', 'alpha_t1', 'alpha_t2', 'alpha_t3',
                 'alpha_022', 'alpha_029', 'alpha_028', 'alpha_021', 'alpha_078', 'alpha_112', 'alpha_028']
positive_filed = ['alpha_t4', 'alpha_111', 'alpha_012', 'alpha_018', 'alpha_019', 'alpha_040', 'alpha_190']

display_field  = ['alpha_t3', 'alpha_018', 'alpha_t2', 'alpha_t1', 'alpha_rsi', 'alpha_022', 'alpha_019', ]
