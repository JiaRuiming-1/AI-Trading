index_range_df = pd.DataFrame(index=universe.index.unique())
for dt in tqdm(universe.index.unique()):
    tmp = universe.loc[universe.index == dt]
    index_range_df.at[dt, 'mean'] = tmp['atr'].mean()
    index_range_df.at[dt, 'std'] = tmp['atr'].std()
index_range_df.head()


def alpha_t7(df):
    def cal_(df):
        benchmark_close = index_df.loc[df.index]['close']
        df['alpha_t7'] = df['close'].rolling(6).apply(
                            lambda y: np.polyfit(benchmark_close.loc[y.index], y, deg=1)[0])
        df['alpha_t7'] = df['alpha_t7'].diff(1)
        return df
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t7(universe)

def alpha_t8(df):
    def cal_(data):
        data['alpha_t8'] = np.where(data['cci']>=150, -data['cci'],
                                   np.where(data['cci']<=-150, data['cci'], data['alpha_t1']))
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t8(universe)


def wins(x,a,b):
    return np.where(x <= a,a, np.where(x >= b, b, x))

for factor_name in tqdm(factor_names):
    #factor_name = 'alpha_wt'
    rolling_obj = universe[factor_name].rolling(len(universe.ts_code.unique()) * 500)
    universe[factor_name] = (universe[factor_name] - rolling_obj.median())/rolling_obj.std()
    universe[factor_name] = wins(universe[factor_name], 
                                 rolling_obj.median() - 3 * rolling_obj.std(), 
                                 rolling_obj.median() + 3 * rolling_obj.std())
    universe[factor_name] = universe[factor_name]/3
    
    
# 生成一些随机数据
x = np.linspace(-5, 5, 50)
y = 4 * x ** 2 - 5* x + 1 + np.random.randn(50) * 5

## + + fast +
## - + slow + ready sell
## + - slow up ready buy
## - -  fast - 

# 用np.polyfit拟合二次函数
p = np.polyfit(x, y, 2)
print(p)
# 绘制原始数据和拟合曲线
plt.scatter(x, y)
plt.plot(x, np.polyval(p, x), 'r')
plt.show()


def alpha_t3(df):
    def quadratic_cal_(x, y):
        x = x.fillna(method='bfill').fillna(.001)
        y = y.fillna(method='bfill').fillna(.001)
        p = np.polyfit(x[y.index], y, deg=2)
        return p[0] * abs(p[1])
    
    def cal_(data):
        close_sma_long = data['close'].rolling(180).mean()
        close_sma_short = data['close'].rolling(6).mean()
        data['close_trend'] = close_sma_short.rolling(14).apply(lambda y: quadratic_cal_(close_sma_long, y))
        data['alpha_t3'] = data['close_trend']
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t3(universe)   


def alpha_t10(df):
    def cal_(data):
        data['alpha_t10'] = data['close'] - (data['close'].rolling(5).mean() - index_range_df['ret_5'])
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t10(universe)
    
time_zscore_factors = [
    'alpha_kama1', 'alpha_atr', 'alpha_atr1', 'alpha_supertrend', 'alpha_supertrend1', 'alpha_t5', 'alpha_t3','alpha_t8','alpha_t9',
    'alpha_075', 'alpha_019'
]

value_zscore_factors = {
    # mean and divide scale
    'alpha_wt': [0, 25],
    'alpha_cci': [0, 360],
    'alpha_cci1':[0, 200],
    'alpha_srsi':[-50, 100],
    'alpha_rsi': [-50, 100],
    'alpha_wr':[50, 100],
    'alpha_ppo':[0, 2],
    'alpha_ppo1':[0, 5],
    'alpha_macd':[0, 0.03],
    'alpha_kdj':[0,75],
    'alpha_kama':[0,0.1],
    'alpha_t1':[0,1],
    'alpha_t2':[0, 0.1],
    'alpha_t2a':[0, 0.1],
    'alpha_t4':[0, 0.1],
    'alpha_t4a':[0, 0.1],
    'alpha_t5a':[0, 0.05],
    'alpha_t6':[0, 0.05],
    'alpha_t7':[0, 200],
    'alpha_t10':[0, 0.1],
}

factor_names = list(value_zscore_factors.keys()) + time_zscore_factors
final_columns = base_columns+factor_names    
        
   
def alpha_t6(df):
    def cal_(data):
        atr_rolling = data['atr'].rolling(180)
        up_line = atr_rolling.mean() + 1.618*atr_rolling.std()
        down_line = atr_rolling.mean() - 1.618*atr_rolling.std()
        data['alpha_t6'] = np.where((data['atr']>=up_line), -data['atr'],
                                   np.where((data['atr']<=down_line), data['atr'], 0))
        data['alpha_t6'] = data['alpha_t6'].rolling(5).mean()
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df

universe = alpha_t6(universe)
