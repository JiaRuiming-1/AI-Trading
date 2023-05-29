def alpha_t1(df):
    def cal_(data):
        data['alpha_t1'] = (data['close'].rolling(4).mean() - data['close'].rolling(20).mean())/data['close'].rolling(20).std()
        data['alpha_t1'] = -data['alpha_t1'] + data['rsi_6'] * 0.2
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t1(universe) 

def alpha_t2(df):
    def cal_(data):
        data['alpha_t2'] = -data['log-ret']/ (data['close'].rolling(6).max() - data['close'].rolling(6).min())
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df
    
universe = alpha_t2(universe)     

def alpha_t3(df):
    def cal_(data):
        data['alpha_t3'] = -data['log-ret'].rolling(4).sum()/(data['high'] - data['low']).rolling(12).sum()
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t3(universe)

import random
universe = universe.sort_values(by=['date','ts_code'])
all_factors = universe.copy(deep=True)
start_time = '2023-01-01 00:00:00'
all_factors = all_factors.loc[all_factors.index>=start_time]
def return_handle(df):
    df['returns_2'] = df['log-ret'].shift(-1)
    noise = abs(random.gauss(0, 0.02))
    if noise > 0.02:
        noise=0.005
    df['returns_2'] = df['returns_2'] + noise*df['vwap'].pct_change().shift(-1).fillna(0.)
    return df
all_factors = all_factors.groupby('ts_code').apply(return_handle)
all_factors = all_factors.replace([np.inf, -np.inf], np.nan).fillna(0.).sort_values(by=['date', 'ts_code'])
print(universe.shape, all_factors.shape)


#'alpha_wt','alpha_rsi', 'alpha_srsi','alpha_t1', 'alpha_t3', 'alpha_029', 'alpha_078',  'alpha_112', 'alpha_131', 'alpha_176',
display_field  = [
   'alpha_wt','alpha_cci', 'alpha_rsi', 'alpha_srsi',  'alpha_078', 'alpha_111', 'alpha_131',
   'alpha_t1', 'alpha_t2',  'alpha_t3', 'alpha_019', 'alpha_t3_fix',
   #'alpha_006', 'alpha_007', 'alpha_008', 'alpha_012', 'alpha_018', 'alpha_029', 'alpha_039', 
   #'alpha_061', 'alpha_112', 'alpha_101', 'alpha_176', 'alpha_t2_fix'
]
