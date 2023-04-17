def factor_positive_profit(universe):
    df = universe.copy(deep=True)
    features = [
       'alpha_atr', 'alpha_010', 'alpha_149', 'alpha_AI'
    ]
    def process_(data):
        ## shift log-ret   
        data['p_ret4'] = data['log-ret'].shift(-2).fillna(0)
        data['p_ret5'] = data['log-ret'].shift(-3).fillna(0)
        data['p_ret6'] = data['log-ret'].shift(-4).fillna(0)
        ## only save factor positive values
        for feature in features:
            data[feature] = np.where(data[feature]>0, data[feature], 0)
        return data
    
    df = df.groupby('ts_code').apply(process_)
    
    ## calculate factor return by shift returns
    for feature in features:
        df[feature + '_returns'] =  (df['p_ret4']  + df['p_ret5'] + df['p_ret6']) * df[feature]/3
    
    ## sum returns by each day
    p_ret_df = pd.DataFrame(index=df.index.unique())
    for dt in tqdm(p_ret_df.index, desc='cross sum feature returns'):
        tmp = df.loc[df.index==dt]
        for feature in features:
            p_ret_df.at[dt, feature] = tmp[feature + '_returns'].sum()
        
    ## calculate positive factor cumsum returns
    p_ret_df.fillna(0)
    for feature in features:
        p_ret_df[feature] = p_ret_df[feature].cumsum()
            
    return p_ret_df

tmp_cal = universe.loc[universe['trade_date'] > 20220101]
positive_returns_df = factor_positive_profit(tmp_cal)
