negative_field = ['alpha_021','alpha_022', 'alpha_029', 'alpha_044', 'alpha_078',  'alpha_089', 
                  'alpha_110', 'alpha_112', 'alpha_128', 'alpha_116', 'alpha_122', 'alpha_131', 'alpha_133', 'alpha_176', ]
universe[negative_field] = -universe[negative_field]

factor_names = [
   'alpha_atr','alpha_wt', 'alpha_wtr', 'alpha_cci', 'alpha_rsi', 'alpha_t1', 'alpha_t3', 'alpha_t5',
   'alpha_019', 'alpha_008', 'alpha_078',  'alpha_018', 'alpha_021', 'alpha_022', 
   'alpha_111', 'alpha_110',   'alpha_131', 'alpha_112', 'alpha_116', 'alpha_128', 'alpha_133',   
   'alpha_035', 'alpha_061','alpha_089', 'alpha_006','alpha_033',  'alpha_039', 'alpha_040', 'alpha_083', 
]

def rescale_rank(data, zscore_features=factor_names):
    data[zscore_features] = data[zscore_features].rank(method='min',pct=True)
    data[zscore_features] = (data[zscore_features] - 0.5) * 2
    return data

bak = universe.copy(deep=True)
start_time = '2022-11-01 00:00:00'
universe = universe.loc[universe.index>=start_time]
universe = my_groupby(universe, 'trade_date', rescale_)

###################################################################################################
q_num = 5
alpha_field  = [
   'alpha_atr','alpha_wt', 'alpha_wtr', 'alpha_cci', 'alpha_rsi', 'alpha_t1', 'alpha_t3', 'alpha_t5',
   'alpha_019', 'alpha_008', 'alpha_078',  'alpha_018', 'alpha_021', 'alpha_022', 
   'alpha_111', 'alpha_110',   'alpha_131', 'alpha_112', 'alpha_116', 'alpha_128', 'alpha_133',   
   'alpha_035', 'alpha_061','alpha_089', 'alpha_033', 'alpha_006', 'alpha_039', 'alpha_040', 'alpha_083', 
]

df = pd.DataFrame(index=alpha_df.index.unique())
for dt in tqdm(alpha_df.index.unique()):
    for feature in alpha_field:
        tmp = alpha_df.loc[alpha_df.index == dt]
        tmp['returns_2'] = wins(tmp['returns_2'], -0.03, 0.03) * (1 - 5e-4)
        df.at[dt, feature] = (tmp['returns_2'] * tmp[feature]).sum()/(tmp[feature].abs().sum()) * 5
        # calculate quartile 3 returns
        try:
            tmp[feature + '_q' + str(q_num)] = pd.qcut(tmp[feature], q=q_num, labels=list(range(1,q_num+1)), duplicates='drop')
        except Exception as e:
            tmp[feature + '_q' + str(q_num)] = 3
        for q in range(1, q_num+1):
            # 1q 2q 3q 4q 5q
            df.at[dt, feature + '_q' + str(q)] = tmp.loc[tmp[feature + '_q'+str(q_num)]==q]['returns_2'].sum()
            
display_field  = [
  #'alpha_wt', 'alpha_atr',  'alpha_cci', 'alpha_wtr', 'alpha_rsi', 'alpha_t1','alpha_t5',
  #'alpha_019', 'alpha_008', 'alpha_078',  'alpha_018', 'alpha_021', 'alpha_022', 
  #'alpha_110', 'alpha_111', 'alpha_131', 'alpha_112', 'alpha_116', 'alpha_128', 'alpha_133',   
  'alpha_035', 'alpha_061','alpha_089', 'alpha_033',  'alpha_039', 'alpha_040', 'alpha_083', 
]

(df[display_field]/5).cumsum().plot()

# alpha_008    7.112507
# alpha_078    5.924321
# alpha_019    5.641981
# alpha_wt     6.093409
# alpha_cci    5.723536
# alpha_035    5.557689

feature = 'alpha_035'
q_df = pd.DataFrame(index = df.index)
for i in range(1, q_num+1):
    q_feature = feature + '_q' + str(i)
    if q_feature in df.columns:
        q_df[q_feature] = (df[q_feature]).cumsum()

q_df.plot()        

            
            
            
            
            
            
            
           
