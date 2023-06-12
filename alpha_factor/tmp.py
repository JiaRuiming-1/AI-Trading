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
