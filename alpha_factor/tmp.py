def rescale_value_factors(universe, zscore_dict = value_zscore_factors):
    for factor_name in tqdm(zscore_dict.keys()):
        rescale_list = zscore_dict[factor_name]
        universe[factor_name] = (universe[factor_name] - rescale_list[0])/rescale_list[1]
        universe[factor_name] = np.where(universe[factor_name].abs()>1, 
                                         np.sign(universe[factor_name]), universe[factor_name])
    return universe
        

universe = rescale_value_factors(universe)


def accuracy_cal(df, feature):
    cond = df[feature] * df['returns_2']
    accuracy2error = df.loc[cond>0].shape[0]/df.loc[cond<0].shape[0]
    accuracy2all = df.loc[cond>0].shape[0]/df.shape[0]
    print(f'{feature}: {accuracy2error}, {accuracy2all}')
    
for feature in factor_names:
    accuracy_cal(alpha_df, feature)


stop_loss_rate = -0.05
cond_up = ((alpha_df['returns_2']>0) & (((alpha_df['open']-alpha_df['low'])/alpha_df['low'])>=abs(stop_loss_rate)))
cond_down = ((alpha_df['returns_2']<0) & (((alpha_df['high']-alpha_df['open'])/alpha_df['open'])>=abs(stop_loss_rate)))
alpha_df.loc[cond_up | cond_down].shape[0] / alpha_df.shape[0]

factor_names = [
    'alpha_cci', 'alpha_kdj',  'alpha_atr', 'alpha_atr1', 'alpha_kama1',  
    'alpha_t4a', 'alpha_t6a', 'alpha_t6b', 'alpha_019'
]

q_num = 5
stop_loss_rate = -0.01
df = pd.DataFrame(index=alpha_df.index.unique())
for dt in tqdm(alpha_df.index.unique()):
    tmp = alpha_df.loc[alpha_df.index == dt]
    tmp['returns_2'] = wins(tmp['returns_2'], -0.06, 0.06)
    for feature in factor_names:
        tmp = keep_top_bottom(tmp, feature, bottom=0.4, top=0.6)
        
        # stop loss conditions
        ret_sr = tmp['returns_2'] * np.sign(tmp[feature])
        cond_up = ((tmp['returns_2']>0) & (((tmp['open']-tmp['low'])/tmp['open'])>abs(0.02)) & (ret_sr>0))
        cond_down = ((tmp['returns_2']<0) & (((tmp['high']-tmp['open'])/tmp['high'])>abs(0.02)) & (ret_sr>0))
        
        # stop loss copy from original forward returns
        tmp['forward_return'] = np.where(cond_up | cond_down, np.sign(tmp['returns_2']) * stop_loss_rate, tmp['returns_2'])
        #tmp['forward_return'] = tmp['returns_2']
        tmp['forward_return'] = np.where(ret_sr<stop_loss_rate, 
                                         np.sign(tmp['returns_2']) * abs(stop_loss_rate), tmp['forward_return'])
        
        # costs
        holding2one_now = tmp[feature]/tmp[feature].abs().sum()
        holding2one_pre = tmp[feature + 'shift1'] / tmp[feature + 'shift1'].abs().sum()
        ret_sr_pre = tmp['log-ret'] * np.sign(tmp[feature + 'shift1'])
        holding2one_pre = np.where(ret_sr_pre<=stop_loss_rate, 0, holding2one_pre)
        
        tmp['holding2one'] = holding2one_now
        costs = (holding2one_now - holding2one_pre).abs().sum() * 5e-4
        liquidate_now_costs = tmp.loc[ret_sr<=stop_loss_rate]['holding2one'].abs().sum() * 5e-4
        
        # factor return
        df.at[dt, feature] = (tmp['forward_return'] * tmp[feature]).sum()/(tmp[feature].abs().sum()) \
                                - costs - liquidate_now_costs     
        # quartile 5 returns
#         try:
#             tmp[feature + '_q' + str(q_num)] = pd.qcut(tmp[feature], q=q_num, 
#                                                        labels=list(range(1,q_num+1)), duplicates='drop')
#         except Exception as e:
#             tmp[feature + '_q' + str(q_num)] = 2
#         for q in range(1, q_num+1):
#             # 1q 2q 3q 4q 5q
#             df.at[dt, feature + '_q' + str(q)] = tmp.loc[tmp[feature + '_q'+str(q_num)]==q]['returns_2'].sum()
