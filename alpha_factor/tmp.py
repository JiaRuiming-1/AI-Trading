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
