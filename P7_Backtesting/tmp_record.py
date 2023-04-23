## calculate risk factor return module
def change_risk_B_col(B):
    new_colnames = ['risk_' + str(_) for _ in B.columns if 'risk_' not in str(_)]
    if len(new_colnames) == 0:
        return B
    B_columns_map = {}
    for i, col in enumerate(B.columns):
        B_columns_map[col] = new_colnames[i]
    B = B.rename(columns=B_columns_map)
    return B

tmp = all_factors.loc[all_factors['trade_date']==20200601].sort_values(by=['ts_code'])
B = change_risk_B_col(B)
B.index.name='ts_code'
tmp = tmp.merge(B.reset_index(), on=['ts_code'], how='left')
estimate_factor_returns(tmp, name='risk_')

## calculate all risk factor returns
for dt in tqdm(calendar, desc='regression risk returns'):
    tmp = all_factors.loc[all_factors['trade_date']==dt].sort_values(by=['ts_code'])
    B = variance_all[dt][1]
    B = change_risk_B_col(B)
    B.index.name='ts_code'
    tmp = tmp.merge(B.reset_index(), on=['ts_code'], how='left')
    facotr_return = estimate_factor_returns(tmp, name='risk_')
    variance_all[dt][2]=facotr_return

variance_all[20200701][2]

## last module
df.at[time_i,"risk.pnl"] = np.sum(rr.values * risk_exposures[dt].values)
