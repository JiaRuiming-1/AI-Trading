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
    
    
        
