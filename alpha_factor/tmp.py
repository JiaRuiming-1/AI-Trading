self.df = self.df.sort_values(by=['date', 'ts_code']).drop_duplicates(subset=['trade_date','ts_code'])

tech_indicator_list = ['log-ret','wt1', 'wt2','stochrsi_5', 'cci_5', 'rsi_5', 'wr_5', 
                       'ppo', 'ppos', 'macds', 'macd', 'kdjj_5', 'kdjk_5', 'kdjd_5', 'close_10_kama_5_30']  #@@@


universe['alpha_wt'] = -(universe['wt1'] - universe['wt2'])
universe['alpha_wt'] = -universe['wt1']
universe['alpha_cci'] = -universe['cci_5']
universe['alpha_srsi'] = -universe['stochrsi_5']
universe['alpha_rsi'] = -universe['rsi_5']
universe['alpha_wr'] = -universe['wr_5']
universe['alpha_ppo'] = (universe['ppos'] - universe['ppo'])
universe['alpha_macd'] = (universe['macds'] - universe['macd'])#@@@
universe['alpha_kdj'] = -(0.6 * (universe['kdjj_5'] - universe['kdjk_5']) + 0.4 * (universe['kdjk_5'] - universe['kdjd_5']))#@@@
universe['alpha_kama2'] = - universe['close_10_kama_5_30'] #@@@


def alpha_kama(df): #@@@
    feature = 'close_10_kama_5_30'
    def cal_(df):
        kama_filter = df[feature].rolling(20).std()
        cond_in1 = (df[feature] - df[feature].shift(4)) >kama_filter
        cond_in2 = (df[feature].shift(4) - df[feature].shift(8)) > kama_filter
        cond_out1 = (df[feature] - df[feature].shift(4)) < -kama_filter
        cond_out2 = (df[feature].shift(4) - df[feature].shift(8)) < -kama_filter
        df['alpha_kama'] = np.where((cond_out1 & cond_out2), -df['log-ret'] , - 1e-3*kama_filter)
        df['alpha_kama'] = np.where((cond_in1 & cond_in2), df['log-ret'] , df['alpha_kama'])
        return df
    
    df_all = pd.DataFrame()
    for ts_code in tqdm(df.ts_code.unique(), desc='alpha_kama processing...'):
        tmp = df.loc[df.ts_code == ts_code]
        tmp = cal_(tmp)
        df_all = df_all.append(tmp)
    return df_all.sort_values(by=['date', 'ts_code'])

universe = alpha_kama(universe)


def rolling_jump(sr, win, interval): #@@@
    result = []
    for i in range(0, len(sr), interval):
        if i + win <= len(sr):
            result.extend([np.nan]*(interval-1) + [sr[i:i+win].mean()])
        else:
            result.extend([np.nan]*(len(sr) % interval))
    print
    return pd.Series(result, sr.index)
  
  
def alpha_t2(df):
    def cal_(data):
        data['buy_percent'] = data['buy_amount']/(data['amount'] - data['buy_amount'])
        data['alpha_t2'] = (data['buy_percent'] * data['log-ret']).rolling(5).sum()
        data['alpha_t2'] = rolling_jump(data['alpha_t2'], 3, 3).fillna(method='ffill')
        return data
    
    df = my_groupby(df, 'ts_code', cal_)
    return df
universe = alpha_t2(universe) 


def alpha_t3(df):
    def cal_(df): # 040
        cond = df['log-ret'].rolling(14).sum() > 0
        df['section1'] = np.where(cond, df['close'].diff(5).diff(4), 0)
        df['section2'] = np.where(~cond, df['close'].diff(5).diff(4), 0)
        df['section1'] = df['section1'].rolling(16).sum()
        df['section2'] = df['section2'].rolling(16).sum()
        df['alpha_t3'] = df['section2'] - df['section1']
        df['alpha_t3'] = rolling_jump(df['alpha_t3'], 3, 3).fillna(method='ffill')
        return df

    df = my_groupby(df, 'ts_code', cal_)
    df = df.drop(columns=['section1', 'section2'])
    return df

universe = alpha_t3(universe)
