factor_names = [
    'alpha_ppo1', 'alpha_macd', 'alpha_t6', 'alpha_atr', 'alpha_atr1', 'alpha_kama1',  
    'alpha_supertrend', 'alpha_t1', 'alpha_t1a',  'alpha_075', 
    'alpha_t8', 'alpha_t1b', 'alpha_t1c', 'alpha_019', 'alpha_t5', 'alpha_cci',
]

q_num = 5
stop_loss_rate = -0.02
df = pd.DataFrame(index=alpha_df.index.unique())
for dt in tqdm(alpha_df.index.unique()):
    tmp = alpha_df.loc[alpha_df.index == dt]
    tmp['returns_2'] = wins(tmp['returns_2'], -0.1, 0.1)
    for feature in factor_names:
        tmp = keep_top_bottom(tmp, feature, bottom=0.4, top=0.6)
        
        # stop loss conditions
        ret_sr = tmp['returns_2'] * np.sign(tmp[feature])
        cond_up = ((tmp['returns_2']>0) & (((tmp['open']-tmp['low'])/tmp['low'])>abs(0.02)) & (ret_sr>0))
        cond_down = ((tmp['returns_2']<0) & (((tmp['high']-tmp['open'])/tmp['open'])>abs(0.02)) & (ret_sr>0))
        
        # stop loss copy from original forward returns
        #tmp['forward_return'] = np.where(cond_up | cond_down, np.sign(tmp['returns_2']) * stop_loss_rate, tmp['returns_2'])
        tmp['forward_return'] = tmp['returns_2']
        tmp['forward_return'] = np.where(ret_sr<stop_loss_rate, 
                                         0, tmp['forward_return'])
        
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
