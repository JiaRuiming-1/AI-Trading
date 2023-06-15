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


stop_loss_rate = -0.04
cond_up = ((alpha_df['returns_2']>0) & (((alpha_df['open']-alpha_df['low'])/alpha_df['low'])>=abs(stop_loss_rate)))
cond_down = ((alpha_df['returns_2']<0) & (((alpha_df['high']-alpha_df['open'])/alpha_df['open'])>=abs(stop_loss_rate)))
alpha_df.loc[cond_up | cond_down].shape[0] / alpha_df.shape[0]

q_num = 5
stop_loss_rate = -0.04
df = pd.DataFrame(index=alpha_df.index.unique())
for dt in tqdm(alpha_df.index.unique()):
    tmp = alpha_df.loc[alpha_df.index == dt]
    tmp['returns_2'] = wins(tmp['returns_2'], -0.1, 0.1)
    for feature in factor_names:
        # stop loss conditions
        ret_sr = tmp['returns_2'] * np.sign(tmp[feature])
        cond_up = ((tmp['returns_2']>0) & (((tmp['open']-tmp['low'])/tmp['low'])>=abs(stop_loss_rate)) & (ret_sr>0))
        cond_down = ((tmp['returns_2']<0) & (((tmp['high']-tmp['open'])/tmp['open'])>=abs(stop_loss_rate)) & (ret_sr>0))
        
        # stop loss copy from original forward returns
        tmp['forward_return'] = np.where(cond_up | cond_down, stop_loss_rate, tmp['returns_2'])
        tmp['forward_return'] = np.where(ret_sr<=stop_loss_rate, stop_loss_rate, tmp['forward_return'])
        
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

['ACHUSDT', 'ADAUSDT', 'APEUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BNBUSDT',
       'BTCUSDT', 'CFXUSDT', 'DOGEUSDT', 'DOTUSDT', 'DYDXUSDT', 'ETHUSDT',
       'FILUSDT', 'FTMUSDT', 'GALAUSDT', 'GMTUSDT', 'LINKUSDT', 'LTCUSDT',
       'MASKUSDT', 'MATICUSDT', 'NEARUSDT', 'SANDUSDT', 'SHIBUSDT',
       'SOLUSDT', 'STXUSDT', 'TRXUSDT', 'UMAUSDT', 'WOOUSDT', 'XRPUSDT']

