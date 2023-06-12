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
