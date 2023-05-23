universe[factor_names] = universe[factor_names]/3.3
universe = universe.sort_values(by=['date','ts_code'])
all_factors = universe.copy(deep=True)
#all_factors = all_factors.sort_values(by=['date'])
def return_handle(df):
    df['returns_2'] = df['log-ret'].shift(-1)
    return df
all_factors = all_factors.groupby('ts_code').apply(return_handle)
all_factors = all_factors.replace([np.inf, -np.inf], np.nan).fillna(0.).sort_values(by=['date', 'ts_code'])
print(universe.shape, all_factors.shape)

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
    estu['returns_2'] = wins(estu['returns_2'], -0.6, 0.6)
    #all_factors = factors_from_names(list(df), name)
    results = pd.Series()
    for factor_name in factor_names:
        form = get_formula([factor_name], "returns_2")
        model = ols(form, data=estu)
        result = model.fit()
        results = results.append(result.params)
    return results
  
test = estimate_factor_returns(all_factors.loc[all_factors['trade_date']=='2022-07-01 00:00:00'])


base_field = ['ts_code', 'log-ret', 'open', 'high', 'low', 'close', 'volume', 'vwap','trade_date']
date_and_code = [ 'ts_code', 'returns_2']

start_time = '2022-07-01 00:00:00'
alpha_df = all_factors[factor_names + date_and_code].copy(deep=True)
alpha_df = alpha_df.loc[alpha_df.index>=start_time]
calendar = alpha_df.index.unique() # int64

#only for positive estimate
# for feature in alpha_field:
#     alpha_df[feature] = np.where(alpha_df[feature]>=0.7, alpha_df[feature], np.where(alpha_df[feature]<=-0.7, alpha_df[feature], 0))
#     alpha_df[feature] = np.where(alpha_df[feature]>0, alpha_df[feature], 0.)

facret = {}
for dt in tqdm(calendar, desc='regression factor returns'):
    facret[dt] = estimate_factor_returns(alpha_df.loc[alpha_df.index==dt])
#facret[calendar[-5]]


date_list = alpha_df.index.unique()
facret_df = pd.DataFrame(index = date_list)

alpha_field  = [
     'alpha_026', 'alpha_028', 'alpha_112', 'alpha_078' , 'alpha_t1', 
     #'alpha_ppo', 'alpha_wt', 'rsi_6', 'alpha_040', 'alpha_srsi',
     #'alpha_019', 'alpha_022', 'alpha_044',  'alpha_128',  'alpha_017', 'alpha_cci'
]



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
