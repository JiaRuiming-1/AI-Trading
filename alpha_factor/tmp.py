universe = feather.read_dataframe('raw30_20220301_20230525.feather')
universe = universe.drop(columns=['close time'])
universe['date'] = pd.to_datetime(universe['date'],format='%Y-%m-%d %H:%M:%S')
universe = universe.set_index(['date']).sort_values(by=['date', 'ts_code'])

# 2023-03-24 20:00:00 data error forward fill
for ts_code in universe.ts_code.unique():
    universe.loc[(universe.ts_code==ts_code) & (universe.index == '2023-03-24 20:00:00')] \
            = universe.loc[(universe.ts_code==ts_code) & (universe.index == '2023-03-24 19:00:00')]
    
universe['vwap'] = universe['amount']/universe['volume']
universe = universe.rename(columns={'open':'open_usdt', 'high':'high_usdt', 'low':'low_usdt', 'close':'close_usdt', 'vwap':'vwap_usdt'})

def my_groupby(df, column, func, sort_keys=['date', 'ts_code']):
    all_df = pd.DataFrame()
    for val in tqdm(df[column].unique()):
        tmp = df.loc[df[column] == val]
        all_df = all_df.append(func(tmp))
    return all_df.sort_index(level=sort_keys)

def convert_price_to_returns(df):
    def cal_(data):
        data['close'] = data['close_usdt'].pct_change()
        data['close'] = (data['close'].fillna(0) + 1).cumprod()
        data['vwap'] = data['vwap_usdt'].pct_change()
        data['vwap'] = (data['vwap'].fillna(0) + 1).cumprod()
        
        for feature in ['open', 'high', 'low',]:
            data[feature] = data[feature + '_usdt']/data['close_usdt'] * data['close']
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = convert_price_to_returns(universe)
universe.loc[(universe.ts_code=='ACHUSDT')][['vwap', 'close']].plot(grid=True)

def alpha_t1(df):
    def cal_(data):
        data['alpha_t1'] = (data['close'].rolling(4).mean() - data['close'].rolling(20).mean())/data['close'].rolling(20).std()
        data['alpha_t1'] = data['alpha_t1'] + data['rsi_6'] * 0.2
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
        data['alpha_t3'] = data['log-ret'].rolling(4).sum()/(data['high'] - data['low']).rolling(12).sum()
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t3(universe)

##############################################################################################
universe[factor_names] = universe[factor_names]/3.3
universe = universe.sort_values(by=['date','ts_code'])

##
universe = universe.sort_values(by=['date','ts_code'])
all_factors = universe.copy(deep=True)
def return_handle(df):
    df['returns_2'] = df['log-ret'].shift(-1)
    return df
all_factors = all_factors.groupby('ts_code').apply(return_handle)
all_factors = all_factors.replace([np.inf, -np.inf], np.nan).fillna(0.).sort_values(by=['date', 'ts_code'])
print(universe.shape, all_factors.shape)
##
def keep_top_bottom(data, bottom=0.25, top=0.75):
    tv = data.quantile(top)
    bv = data.quantile(bottom)
    data = pd.Series(np.where(data>=tv, data, np.where(data<=bv, data, 0)), index=data.index)
    return data
##
def wins(x,a,b):
    return np.where(x <= a,a, np.where(x >= b, b, x))

def get_formula(factors, Y):
    L = ["0"]
    L.extend(factors)
    return Y + " ~ " + " + ".join(L)

def factors_from_names(n, name):
    return list(filter(lambda x: name in x, n))

def estimate_factor_returns(df, name='alpha_'): 
    ## winsorize returns for fitting 
    estu = df.copy(deep=True)
    estu['returns_2'] = wins(estu['returns_2'], -0.2, 0.2)
    #all_factors = factors_from_names(list(df), name)
    results = pd.Series()
    for factor_name in factor_names:
        form = get_formula([factor_name], "returns_2")
        model = ols(form, data=estu)
        result = model.fit()
        results = results.append(result.params)
    return results

test = estimate_factor_returns(all_factors.loc[all_factors['trade_date']=='2022-05-01 00:00:00'])

##
alpha_field  = [
    'alpha_wt',  'alpha_cci', 'alpha_srsi', 'alpha_t2',  
    'alpha_009', 'alpha_016',  'alpha_028', 'alpha_078', 'alpha_112',
]

base_field = ['ts_code', 'log-ret', 'open', 'high', 'low', 'close', 'volume', 'vwap','trade_date']
date_and_code = [ 'ts_code', 'returns_2']

start_time = '2022-05-01 00:00:00'
alpha_df = all_factors[factor_names + date_and_code].copy(deep=True)
alpha_df = alpha_df.loc[alpha_df.index>=start_time]
calendar = alpha_df.index.unique() # int64

for feature in alpha_field:
    alpha_df[feature] = keep_top_bottom(alpha_df[feature])

## evaluate method 1
facret = {}
for dt in tqdm(calendar, desc='regression factor returns'):
    facret[dt] = estimate_factor_returns(alpha_df.loc[alpha_df.index==dt])


date_list = alpha_df.index.unique()
facret_df = pd.DataFrame(index = date_list)

for ii, dt in zip(calendar,date_list): 
    for alp in alpha_field: 
        facret_df.at[dt, alp] = facret[ii][alp]

for column in facret_df.columns:
    plt.plot(facret_df[column].cumsum(), label=column)
    #plt.plot(facret_df[column], label=column)
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('Cumulative Factor Returns')
plt.show()

## evaluate method 2
df = pd.DataFrame(index=alpha_df.index.unique())
for dt in tqdm(alpha_df.index.unique()):
    for feature in alpha_field:
        tmp = alpha_df.loc[alpha_df.index == dt]
        df.at[dt, feature] = (tmp['returns_2'] * tmp[feature]).sum()/(tmp[feature].abs().sum()) * 5

##
display_field  = [
    'alpha_wt',  'alpha_cci', 'alpha_srsi', 'alpha_t2',  
    'alpha_009', 'alpha_016',  'alpha_028', 'alpha_078', 'alpha_112',
    
]
df[display_field].cumsum().plot()

# sharp ratio
np.sqrt(6*252) * df[display_field].mean()/ df[display_field].std()

