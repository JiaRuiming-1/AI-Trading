import pandas as pd
import numpy as np
from scipy.stats import zscore, spearmanr, rankdata
from statsmodels.formula.api import ols
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')


def benchmark_close_day(df):
    # calculate one day benchmark close values
    df['weights'] = df['total_mv'] / df['total_mv'].sum()
    return (df['weights'] * df['close']).sum()


def benchmark_open_day(df):
    # calculate one day benchmark close values
    df['weights'] = df['total_mv'] / df['total_mv'].sum()
    return (df['weights'] * df['open']).sum()


def Getfama3factors(df, benchmark_return):
    def cal_(df):
        SMB = (df.loc[df['rsize'] <= 0.1]['log-ret'].sum() - df.loc[df['rsize'] >= 0.9]['log-ret'].sum()) / 3
        HML = (df.loc[df['rvalue'] <= 0.3]['log-ret'].sum() - df.loc[df['rvalue'] >= 0.7]['log-ret'].sum()) / 2
        return SMB, HML

    df[['rsize', 'rvalue']] = df.groupby('trade_date')[['total_mv', 'pb']].rank(method='min', pct=True)
    fama_df = pd.DataFrame(index=df.index.unique())
    for dt in tqdm(fama_df.index):
        tmp = df.loc[df.index == dt]
        SMB, HML = cal_(tmp)
        fama_df.at[dt, 'SMB'] = SMB
        fama_df.at[dt, 'HML'] = HML
        # fama_df.at[dt, 'MKT'] = tmp['log-ret'].mean() - benchmark_return[dt]
        fama_df.at[dt, 'MKT'] = tmp['log-ret'].mean() - 0.00015

    return fama_df.fillna(0.)


def Corr(data, win_len):
    obj = data.rolling(window=win_len, method='table')
    s = []
    for o in obj:
        if o.shape[0] < win_len:
            s.append(0.)
        else:
            s.append(spearmanr(o.iloc[:, 0], o.iloc[:, 1])[0])
            # s.append(o.iloc[:, 0].corr(o.iloc[:, 1]))
    return pd.DataFrame(s, index=data.index, columns=['corr'])


def Decaylinear(sr, window):
    weights = np.array(range(1, window + 1))
    sum_weights = np.sum(weights)
    return sr.rolling(window).apply(lambda x: np.sum(weights * x) / sum_weights)


## calculate series rank in a rolling period time and get last time rank value
def Tsrank(sr, window):
    return sr.rolling(window).apply(lambda x: rankdata(x)[-1])


def Sma(sr, n, m):
    return sr.ewm(alpha=m / n, adjust=False).mean()


def Sequence(n):
    return np.arange(1, n + 1)


def Lowday(sr, window):
    return sr.rolling(window).apply(lambda x: len(x) - x.values.argmin())


def Highday(sr, window):
    return sr.rolling(window).apply(lambda x: len(x) - x.values.argmax())


def Regbeta(sr, x):
    window = len(x)
    return sr.rolling(window).apply(lambda y: np.polyfit(x, y, deg=1)[0])


def RegResi(df, fama_df, form='ret ~ SMB + HML + MKT'):
    # pd.DataFrame.merge()
    estu = fama_df.loc[df.index].copy()
    estu = df.merge(estu, on=['date'])
    estu = estu.rename(columns={'log-ret': 'ret'})
    model = ols(form, data=estu)
    results = model.fit()
    return results.params.Intercept


def rescale_zscore(data):
    for col in data.columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data


class Alpha191():
    def __init__(self, df, benchmark_df=None):
        df = df.reset_index(drop=True)
        df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index(['date']).sort_values(by=['date'])

        self.benchmark_close = df.groupby('date').apply(benchmark_close_day)
        self.benchmark_open = df.groupby('date').apply(benchmark_open_day)
        self.benchmark_return = self.benchmark_close.pct_change().fillna(0.)

        if benchmark_df is not None:
            benchmark_df = benchmark_df.reset_index(drop=True)
            benchmark_df['date'] = pd.to_datetime(benchmark_df['trade_date'], format='%Y%m%d')
            benchmark_df = benchmark_df.set_index(['date']).sort_values(by=['date'])
            self.benchmark_close = benchmark_df['close']
            self.benchmark_open = benchmark_df['open']
            self.benchmark_return = benchmark_df['pct_chg']

        self.df = df
        self.fama_df = Getfama3factors(self.df, self.benchmark_return)

    ## need nothing (good based on large tickers pool)
    def alpha001(self, df):
        ## 6日成交量 与 涨幅关系
        ##### (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))####
        def cal_(df):
            df['section1'] = np.log(df['volume'])
            ##====improve====
            # df['section1'] = df['volume']/df['total_share']
            df['section1'] = df['section1'].diff(1)
            return df

        df = df.groupby('ts_code').apply(cal_)
        df['section2'] = ((df.close - df.open) / df.open)
        df[['section1', 'section2']] = df.groupby('trade_date')[['section1', 'section2']].rank(method='min', pct=True)
        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='alpha001 processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp['alpha_001'] = -Corr(tmp[['section1', 'section2']], 6)['corr']
            df_all = df_all.append(tmp)
        df_all = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        self.df = df_all
        return self.df

    ## need zscore and percent (good)
    def alpha002(self, df):
        ## K线形态变化
        ##### -1 * delta((((close-low)-(high-close))/(high-low)),1))####
        df['alpha_002'] = -(((df.close - df.low) - (df.high - df.close)) / (df.high - df.low)).diff(1)
        ##===improve====
        df['alpha_002'] = df.groupby('trade_date')['alpha_002'].rank(method='min', pct=True)
        self.df = df
        return self.df

    ## need zscore and percent
    def alpha003(self, df):
        ## 累计阶段涨跌趋势
        ##### SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6) ####
        def cal_by_ticker(df):
            cond1 = (df.close == df.close.shift(1).fillna(method='ffill'))
            cond2 = (df.close > df.close.shift(1).fillna(method='ffill'))
            cond3 = (df.close < df.close.shift(1).fillna(method='ffill'))
            part = df.close.copy(deep=True)
            part[cond1] = 0
            part[cond2] = df.close - np.minimum(df.low, df.close.shift(1).fillna(method='ffill'))
            part[cond3] = df.close - np.maximum(df.high, df.close.shift(1).fillna(method='ffill'))
            df['alpha_003'] = part.rolling(window=6).sum()
            ## ===improve===
            df['alpha_003'] = df['alpha_003'] / df['close']
            return df

        self.df = df.groupby('ts_code').apply(cal_by_ticker)
        return self.df

    ## need zscore and percent (no)
    def alpha004(self, df_all):
        ## 短期突破类似cci
        #####((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
        def cal_(df):
            cond_L = df['close'].rolling(window=8).apply(lambda x: x.mean() + x.std())
            cond_R = df['close'].rolling(window=2).mean()
            cond1 = cond_L < cond_R
            cond2 = cond_L > cond_R
            cond3 = cond_L == cond_R
            cond4 = (df['volume'] / df['volume'].rolling(window=20).mean()) >= 1
            part = df.close.copy(deep=True)
            part[cond1] = -1
            part[cond2] = 1
            part[cond3] = -1
            part[cond3 & cond4] = 1
            df['alpha_004'] = part
            ## ===improve===
            df['alpha_004'] = df['alpha_004'] / df['close']
            return df

        self.df = df_all.groupby('ts_code').apply(cal_)
        return self.df

    ## need nothing
    def alpha005(self, df):
        ## 5日最大价量相关系数
        ####(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))###
        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='alpha005 processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp['Tsrank_v'] = Tsrank(tmp.volume, 5)
            tmp['Tsrank_h'] = Tsrank(tmp.high, 5)
            tmp['alpha_005'] = Corr(tmp[['Tsrank_v', 'Tsrank_h']], 5)['corr']
            tmp['alpha_005'] = -tmp['alpha_005'].rolling(window=3).max()
            df_all = df_all.append(tmp)
        df_all['alpha_005'].fillna(0., inplace=True)
        self.df = df_all.drop(columns=['Tsrank_v', 'Tsrank_h']).sort_values(by=['date'])
        return self.df

    ## need zscore and percent (test)
    def alpha006(self, df_all):
        ## 开盘半小时寻找买卖点
        ####(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)###
        def cal_(df):
            df['alpha_006'] = -np.sign(((df.open * 0.85) + (df.high * 0.15)).diff(4).fillna(0.))
            return df

        df_all = df_all.groupby('ts_code').apply(cal_)
        df_all['alpha_006'] = df_all.groupby('trade_date')['alpha_006'].rank(method='min', pct=True)
        self.df = df_all
        return self.df

    ## execllent
    def alpha007(self, df):
        ## 横截面价量操作方式（具备买卖方向）
        ####((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))###
        def cal_(df):
            rolling_obj = (df['vwap'] - df['close']).rolling(3)
            df['section1'] = rolling_obj.max()
            df['section2'] = rolling_obj.min()
            df['section3'] = df['volume'].diff(3)
            return df

        df_all = df.groupby('ts_code').apply(cal_)
        df_all[['section1', 'section2', 'section3']] = df_all.groupby('trade_date')[
            ['section1', 'section2', 'section3']] \
            .rank(method='min', pct=True)
        df_all['alpha_007'] = (df_all['section1'] + df_all['section2']) * df_all['section3']
        self.df = df_all.drop(columns=['section1', 'section2', 'section3'])
        return self.df

    # need nothing (test)
    def alpha008(self, df):
        ## 4日价差突破波动逆转
        ####RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)###
        # df['alpha_008'] = ((df['high'] + df['low']) * 0.2 + df['vwap'] * 0.8)
        ### ====improve====
        df['alpha_008'] = ((df['high'] + df['low']) * 0.2 + df['vwap'] * 0.8)
        df['alpha_008'] = - df.groupby('ts_code')['alpha_008'].diff(4)
        df['alpha_008'] = df.groupby('trade_date')['alpha_008'].rank(method='min', pct=True)
        self.df = df
        return self.df

    # need percent (test)
    def alpha009(self, df):
        ## 价量背离（钝化）
        ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)###
        def cal_(df):
            term = ((df.high + df.low) / 2 - (df.high.diff(1) + df.low.diff(1)) / 2) * (df.high - df.low) / df.volume
            df['alpha_009'] = Sma(term, 7, 2)
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    # excellent
    def alpha010(self, df):
        ## 横截面相对价格突破
        ####(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))###
        def cal_(df):
            df['alpha_010'] = np.where(df['log-ret'] < 0, df['log-ret'].rolling(20).std(), df['close'])
            df['alpha_010'] = (df['alpha_010'] ** 2).rolling(window=6).max()
            return df

        df = df.groupby('ts_code').apply(cal_)
        #df['alpha_010'] = df.groupby('trade_date')['alpha_010'].rank(method='min', pct=True)
        df['alpha_010'] = -df['alpha_010']
        self.df = df
        return self.df

    # need percent (no)
    def alpha011(self, df):
        # 价量背离（原始简易构造）
        ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)###
        def cal_(df):
            df['alpha_011'] = ((df.close - df.low) * 1000 / ((df.high - df.low) * df.volume)).rolling(6).sum()
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    # need add offset (good)
    def alpha012(self, df):
        # 大幅低开
        ####(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))###
        def cal_(df):
            df['section1'] = df.open - df.vwap.rolling(window=10).mean()
            df['section2'] = (df.close - df.vwap).abs()
            return df

        df = df.groupby('ts_code').apply(cal_)
        df[['section1', 'section2']] = df.groupby('trade_date')[['section1', 'section2']].rank(method='min', pct=True)
        df['alpha_012'] = df['section1'] * (df['section2'])

        self.df = df.drop(columns=['section1', 'section2'])
        return self.df

    # need zscore and percent(no)
    def alpha013(self, df):
        ## 感觉没什么用
        ####(((HIGH * LOW)^0.5) - VWAP)###
        ## ===improve====
        df['alpha_013'] = (((df.high * df.low) ** 0.5) - df.vwap) / df.vwap
        self.df = df
        return self.df

    # need zscore and percent (no)
    def alpha014(self, df):
        ## 动量预判
        ####CLOSE-DELAY(CLOSE,5)###
        def cal_(df):
            ## ==== improve ====
            df['alpha_014'] = (df['close'] - df['close'].diff(5)) / df['close']
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    # need percent (no)
    def alpha015(self, df):
        ## 隔夜收益率
        ####OPEN/DELAY(CLOSE,1)-1###
        def cal_(df):
            df['alpha_015'] = (df['open'] - df['close'].shift(1).fillna(method='ffill')) / df['open']
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    # excellent
    def alpha016(self, df):
        ## 5日横截面最大价量负相关
        ####(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))###
        df[['section1', 'section2']] = df.groupby('trade_date')[['volume', 'vwap']].apply(zscore)
        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='alpha016 processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp['alpha_016'] = Corr(tmp[['section1', 'section2']], 6)['corr']
            df_all = df_all.append(tmp)

        def cal_(df):
            df['alpha_016'] = df['alpha_016'].rolling(6).max()
            df['alpha_016'] = -df['alpha_016']
            return df

        df_all['alpha_016'] = df_all.groupby('trade_date')['alpha_016'].rank(method='min', pct=True)
        df_all = df_all.groupby('ts_code').apply(cal_)
        self.df = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        return self.df

    # need nothing (good)
    def alpha017(self, df):
        ## 15日以来的最新上涨动力
        ####RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)###
        def cal1_(df):
            df['alpha_017'] = (df['vwap'] - df['vwap'].rolling(window=15).max())
            return df

        def cal2_(df):
            df['alpha_017'] = -df['alpha_017'] ** (df['close'].diff(5))
            return df

        df = df.groupby('ts_code').apply(cal1_)
        df['alpha_017'] = df.groupby('trade_date')['alpha_017'].rank(method='min', pct=True)
        self.df = df.groupby('ts_code').apply(cal2_)
        return self.df

    # need nothing (no)
    def alpha018(self, df):
        # 5日涨跌动力 （简易）
        ####CLOSE/DELAY(CLOSE,5)###
        def cal_(df):
            # df['alpha_018'] = df['close'] / df['close'].shift(5)
            # ==== improve ====
            df['alpha_018'] = -df['close'] / df['close'].shift(5)
            return df

        df = df.groupby('ts_code').apply(cal_)
        # ==== improve ====
        df['alpha_018'] = df.groupby('trade_date')['alpha_018'].rank(method='min', pct=True)
        self.df = df
        return self.df

    # need percent
    def alpha019(self, df):
        # 5日涨跌动力 (归一化)
        ####(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))###
        def cal_(df):
            df['alpha_019'] = np.where(df['close'] < df['close'].shift(5),
                                       (df['close'] - df['close'].shift(5)) / df['close'].shift(5),
                                       np.where(df['close'] > df['close'].shift(5),
                                                (df['close'] - df['close'].shift(5)) / df['close'], 0.))
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    # need nothing (test)
    def alpha020(self, df):
        ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100###
        # 6日涨跌动力 （简易）
        ####CLOSE/DELAY(CLOSE,5)###
        def cal_(df):
            df['alpha_020'] = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)
            return df

        self.df = df.groupby('ts_code').apply(cal_)

        return self.df

    # excellent
    def alpha021(self, df):
        # 14日close均值~时间窗口回归
        ####REGBETA(MEAN(CLOSE,6),SEQUENCE(6))###
        def cal_(df):
            df['alpha_021'] = Regbeta(df['close'].rolling(14).mean(), Sequence(14))
            df['alpha_021'] = - df['alpha_021']
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='alpha021 processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        self.df = df_all.sort_values(by=['date'])
        return self.df

    # need percent (test)
    def alpha022(self, df):
        ## 6日收益均值趋势 12日sma平滑处理
        ####SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)###
        def cal_(df):
            section1 = (df['close'] - df['close'].rolling(window=6).mean()) / df['close'].rolling(window=6).mean()
            section2 = section1.shift(3)
            df['alpha_022'] = Sma((section1 - section2), 12, 1)
            return df

        df = df.groupby('ts_code').apply(cal_)
        self.df = df.sort_values(by=['date'])
        return self.df

    ## percent(no)
    def alpha023(self, df):
        ## 20日上涨波动大小与收益正相关
        ####SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / (SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) + SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100###
        def cal_(df):
            cond = df['close'] > df['close'].shift(1)
            part1 = df.close.copy(deep=True)
            part1[cond] = df['close'].rolling(window=20).std()
            part1[~cond] = 0
            part2 = df.close.copy(deep=True)
            part2[~cond] = df['close'].rolling(window=20).std()
            part2[cond] = 0
            df['alpha_023'] = 100 * Sma(part1, 20, 1) / (Sma(part1, 20, 1) + Sma(part2, 20, 1))
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    def alpha024(self, df):
        ####SMA(CLOSE-DELAY(CLOSE,5),5,1)###
        def cal_(df):
            df['alpha_024'] = Sma((df['close'] - df['close'].shift(5)) / df['close'], 5, 1)
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    # need percent (test)
    def alpha025(self, df):
        ## 量价背离 + 强者恒强
        ####((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))###
        def cal_(df):
            df['section1'] = df['close'].diff(7)
            df['section2'] = Decaylinear((df['volume'] / df['volume'].rolling(20).mean()), 9)
            df['section3'] = df['log-ret'].rolling(120).sum()
            return df

        df = df.groupby('ts_code').apply(cal_)
        df[['section2', 'section3']] = df.groupby('trade_date')[['section2', 'section3']].rank(method='min', pct=True)
        df['section1'] = df['section1'] * (1 - df['section2'])
        df[['section1']] = df.groupby('trade_date')[['section1']].rank(method='min', pct=True)
        # ===improve===
        #  df['alpha_025'] = (-df['section1']) * (1 + df['section3'])
        df['alpha_025'] = ((-df['section1']) * (1 + df['section3']))  # .fillna(-1.) + .8
        self.df = df.drop(columns=['section1', 'section2', 'section3'])
        return self.df

    # need percent
    def alpha026(self, df):
        ## 长线超跌买入
        ####((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))###
        def cal_(df):
            df['section1'] = (df['close'].rolling(7).mean() - df['close'])
            df['section2'] = df['close'].shift(5)
            df['section2'] = Corr(df[['vwap', 'section2']], 120)
            df['alpha_026'] = df['section1'] + df['section2']
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='alpha026 processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)
        self.df = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        return self.df

    ## need nothing (no)
    def alpha027(self, df):
        ## 平滑回报评估
        ####WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)###
        def cal_(df):
            con1 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
            con2 = (df['close'] - df['close'].shift(6)) / df['close'].shift(6) * 100
            df['alpha_027'] = (con1 + con2) / 2
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    ## need percent (no)
    def alpha028(self, df):
        ## 3日夏普比率涨跌趋势
        ####3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)###
        def cal_(df):
            con1 = 3 * Sma((df['close'] - df['low'].rolling(9).min()) / (df['high'].rolling(9).max()
                                                                         - df['low'].rolling(9).min()), 3, 1)
            con2 = 2 * Sma(con1, 3, 1)
            df['alpha_028'] = con1 - con2
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    ## need nothing (test)
    def alpha029(self, df):
        ## 收益率换手率反比关系
        ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME###
        def cal_(df):
            df['alpha_029'] = (df['close'] - df['close'].shift(6)) / (df['close'].shift(6) * df['volume'])
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    # need percent (excellent)
    def alpha030(self, df):
        ## 三因子回归偏离度
        ####WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML， 60))^2,20)###
        def cal_(df):
            obj = df.rolling(window=60, method='table')
            s = []
            for o in obj:
                s.append(RegResi(o, self.fama_df) ** 2)
            sdf = pd.DataFrame(s, index=df.index, columns=['beta'])
            #sdf['beta'] = sdf['beta'] / sdf['beta'].rolling(60).max()
            return sdf

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='alpha030 processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp['alpha_030'] = -cal_(tmp)['beta']
            df_all = df_all.append(tmp)
        #df_all['alpha_030'] = df_all.groupby('trade_date')[['alpha_030']].rank().apply(zscore)
        self.df = df_all
        return self.df

    # excellent
    def alpha032(self, df):
        ## 找14日背离关系最大
        ####(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))###

        df['high_r'] = df['revenue_ps'] / df['close']
        # df['volume_r'] = df['volume']/df['total_share']
        # df[['high_r', 'volume_r']] = df.groupby('ts_code')[['high_r', 'volume_r']].rank(method='min', pct=True)
        df['high_r'] = df.groupby('ts_code')['high_r'].rank(method='min', pct=True)
        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp['alpha_032'] = Corr(tmp[['high_r', 'volume']], 14)['corr']
            df_all = df_all.append(tmp)
        #df_all['alpha_032'] = df_all.groupby('trade_date')['alpha_032'].rank(method='min', pct=True)
        # df_all['alpha_032'] = df_all['alpha_032'].rolling(window=3).sum()
        self.df = df_all.drop(columns=['high_r']).sort_values(by=['date'])
        return self.df


    def alpha033(self, df):
        #### 寻找横截面相距年收益均值超跌加缩量
        ####((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))###
        def cal_(df):
            section1 = df['low'].rolling(6).min()
            df['section1'] = ((-section1) + section1.shift(6))
            df['section2'] = (df['log-ret'].rolling(90).sum() - df['log-ret'].rolling(20).sum()) / 120
            df['section3'] = Tsrank(df['volume'], 6)
            return df

        df = df.groupby('ts_code').apply(cal_)
        df['section2'] = df.groupby('trade_date')['section2'].rank(method='min', pct=True)
        df['alpha_033'] = df['section1'] * df['section2'] * df['section3']
        #df['alpha_033'] = df.groupby('trade_date')['alpha_033'].rank(method='min', pct=True)
        self.df = df.drop(columns=['section1', 'section2', 'section3']).sort_values(by=['date'])
        return self.df

    # need nothing (good)
    def alpha035(self, df):
        ## 寻找开盘价长期高点或量价背离最大化取保守值
        ####(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +(OPEN *0.35)), 17),7))) * -1)###
        def cal_(df):
            df['section1'] = Decaylinear(df['open'].diff(1), 15)
            df['section2'] = df['open'] * 0.65 + df['high'] * 0.35
            df['section2'] = Decaylinear(Corr(df[['volume', 'section2']], 17)['corr'], 7)
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        df_all[['section1', 'section2']] = df_all.groupby('trade_date')[['section1', 'section2']].rank(pct=True)
        df_all['alpha_035'] = -np.where(df_all['section1'] < df_all['section2'], df_all['section1'], df_all['section2'])
        self.df = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        return self.df

    # need nothing (good)
    def alpha039(self, df):
        ## 长线寻找价差小且量大
        ####((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)###
        def cal_(df):
            df['section1'] = Decaylinear(df['close'].diff(2), 8)
            df['section2'] = df['vwap'] * 0.3 + df['open'] * 0.7
            df['section3'] = (df['volume'].rolling(180).mean()).rolling(37).sum()
            df['section2'] = Decaylinear(Corr(df[['section2', 'section3']], 14), 12)
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)
        df_all[['section1', 'section2']] = df_all.groupby('trade_date')[['section1', 'section2']].rank(method='min',
                                                                                                       pct=True)
        df_all['alpha_039'] = df_all['section2'] - df_all['section1']
        self.df = df_all.drop(columns=['section1', 'section2', 'section3']).sort_values(by=['date'])

        return self.df

    #execllent
    def alpha040(self, df):
        ## 26日 涨跌成交量比
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100###
        def cal_(df):
            cond = df['close'] > df['close'].shift(1)
            df['section1'] = np.where(cond, df['volume'], 0)
            df['section2'] = np.where(cond, 0, df['volume'])
            df['section1'] = df['section1'].rolling(26).sum()
            df['section2'] = df['section2'].rolling(26).sum()
            return df

        df = df.groupby('ts_code').apply(cal_)
        df['alpha_040'] = -df['section1'] / (df['section2'])
        #df['alpha_040'] = df.groupby('trade_date')['alpha_040'].rank(method='min', pct=True)
        self.df = df.drop(columns=['section1', 'section2'])
        return self.df

    ## need cross rank
    def alpha044(self, df):
        ## 加权平均价中期价量关系
        ####(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15))###
        def cal_(df):
            df['section1'] = df['volume'].rolling(window=10).mean()
            df['section1'] = Decaylinear(Corr(df[['low', 'section1']], 7)['corr'], 6)
            df['section1'] = Tsrank(df['section1'], 4)
            df['section2'] = Tsrank(Decaylinear(df['vwap'].diff(3), 10), 15)
            df['alpha_044'] = 0.6 * df['section1'] + 0.4 * df['section2']
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        self.df = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        return self.df

    # need nothing
    def alpha056(self, df):
        ## 寻找12日开盘价最小且量价背离越明显的位置
        ####(RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),SUM(MEAN(VOLUME,40), 19), 13))^5)))###
        def cal_(df):
            df['section1'] = df['open'] - df['open'].rolling(12).min()
            df['section2'] = ((df['high'] + df['low']) / 2).rolling(19).sum()
            df['section3'] = ((df['volume'].rolling(40).mean()).rolling(19).sum())
            df['section2'] = Corr(df[['section2', 'section3']], 13)['corr']
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)
        df_all['section2'] = df_all.groupby('trade_date')['section2'].rank(method='min', pct=True)
        df_all['section2'] = df_all['section2'] ** 5
        df_all['section2'] = df_all.groupby('trade_date')['section2'].rank(method='min', pct=True)
        df_all['alpha_056'] = np.where(df_all['section1'] < df_all['section2'], 1, 0)
        self.df = df_all.drop(columns=['section1', 'section2', 'section3']).sort_values(by=['date'])
        return self.df

    # need nothing
    def alpha061(self, df):
        ## 寻找12日均价价最小且80日均量价背离越明显的位置
        ####(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)###
        def cal_(df):
            df['section1'] = -Decaylinear(df['vwap'].diff(1) / df['vwap'], 12)
            df['section2'] = df['volume'].rolling(80).mean()
            df['section2'] = -Corr(df[['low', 'section2']], 8)['corr']
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)
        df_all[['section1', 'section2']] = df_all.groupby('trade_date')[['section1', 'section2']].rank(method='min',
                                                                                                       pct=True)
        df_all['section2'] = Decaylinear(df_all['section2'], 17)
        df_all['section2'] = df_all.groupby('trade_date')['section2'].rank(method='min', pct=True)
        df_all['alpha_061'] = np.where(df_all['section1'] > df_all['section2'], df_all['section1'], df_all['section2'])
        self.df = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        return self.df

    # need nothing (good)
    def alpha069(self, df):
        ## 20日开盘价差动量
        ####(SUM(DTM,20)>SUM(DBM,20)？ (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)： (SUM(DTM,20)=SUM(DBM,20)？0： (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))###
        ####DTM (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
        ####DBM (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
        def cal_(df):
            cond1 = (df['open'] <= df['open'].shift(1))
            df['DTM'] = np.where(cond1, 0, np.maximum(df['high'] - df['open'], df['open'] - df['open'].shift(1)))
            df['DBM'] = np.where(cond1, np.maximum(df['high'] - df['open'], df['open'] - df['open'].shift(1)), 0)
            section1 = df['DTM'].rolling(20).sum()
            section2 = df['DBM'].rolling(20).sum()
            df['alpha_069'] = np.where(section1 == section2, 0, (section1 - section2) / section1)
            df['alpha_069'] = np.where(section1 < section2, (section1 - section2)/section2, df['alpha_069'])
            df['alpha_069'] = - df['alpha_069']
            return df

        df = df.groupby('ts_code').apply(cal_)
        self.df = df.drop(columns=['DTM', 'DBM'])
        return self.df

    ## need nothing (excellent)
    def alpha075(self, df):
        ## (改造)与指数偏离度正相关
        ####COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)###
        def cal_(df):
            benchmark_close = self.benchmark_close[df.index]
            benchmark_open = self.benchmark_open[df.index]
            benchmark_return = self.benchmark_return[df.index]
            cond1 = ((df['log-ret']<0.01) & (benchmark_return > 0))
            cond2 = ((df['log-ret']<0.01) & (benchmark_close > benchmark_open))
            df['section1'] = np.where(cond1, 1, 0)
            df['section2'] = np.where(cond2, 1, 0)
            df['alpha_075'] = df['section1'].rolling(50).sum() + df['section2'].rolling(50).sum()
            return df

        df = df.groupby('ts_code').apply(cal_)
        df['alpha_075'] = df.groupby('trade_date')['alpha_075'].rank(method='min', pct=True)
        self.df = df.drop(columns=['section1', 'section2'])
        return self.df

    ## excellent
    def alpha077(self, df):
        ## 短线价量相关性
        #### MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))###
        def cal_(df):
            df['section1'] = (df['high'] + df['low']) / 2 - df['vwap']
            df['section1'] = Decaylinear(df['section1'], 20)
            section2_1 = (df['high'] + df['low']) / 2
            section2_2 = df['volume'].rolling(40).mean()
            tmp = pd.concat([section2_1, section2_2], axis=1)
            df['section2'] = Decaylinear(Corr(tmp, 6)['corr'], 14)
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        df_all[['section1', 'section2']] = df_all.groupby('trade_date')[['section1', 'section2']].rank(method='min',
                                                                                                       pct=True)
        df_all['alpha_077'] = np.where(df_all['section1'] < df_all['section2'], df_all['section1'], df_all['section2'])
        self.df = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])

        return self.df

    # need zscore and percent
    def alpha078(self, df):
        ####((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))###
        def cal_(df):
            section = (df['high'] + df['low'] + df['close']) / 3
            df['alpha_078'] = (section - section.rolling(12).mean()) / \
                              (abs(df['close'] - section.rolling(12).mean())).rolling(12).mean() / 0.015
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df

    ## need nothing
    def alpha083(self, df):
        ## 价量协方差负相关
        ####(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))###
        def cal_(df):
            df['alpha_083'] = -df['high_r'].rolling(5).cov(df['volume_r'])
            return df

        df[['high_r', 'volume_r']] = df.groupby('trade_date')[['high', 'volume']].rank(method='min', pct=True)
        df = df.groupby('ts_code').apply(cal_)
        #df['alpha_083'] = df.groupby('trade_date')['alpha_083'].rank(method='min', pct=True)
        self.df = df.drop(columns = ['high_r', 'volume_r'])
        return self.df

    ## need nothing (excellent)
    def alpha089(self, df):  # 1797
        ####2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))###
        def cal_(df):
            df['alpha_089'] = 2 * (Sma(df['close'], 13, 2) - Sma(df['close'], 27, 2)
                                   - Sma(Sma(df['close'], 13, 2) - Sma(df['close'], 27, 2), 10, 2)) \
                              / Sma(df['close'], 13, 2)
            return df

        df = df.groupby('ts_code').apply(cal_)
        df['alpha_089'] = df.groupby('trade_date')['alpha_089'].rank(method='min', pct=True)
        self.df = df
        return self.df

    ## need nothing
    def alpha092(self, df):  # 1786
        ####(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)###
        def cal_(df):
            df['section1'] = df['close'] * 0.35 + df['vwap'] * 0.65
            df['section1'] = Decaylinear(df['section1'].diff(2) / df['section1'], 3)
            section2 = pd.concat([df['volume'].rolling(180).mean(), df['close']], axis=1)
            section2 = abs(Corr(section2, 13)['corr'])
            df['section2'] = Tsrank(Decaylinear(section2, 5), 15)
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        df_all['section1'] = df_all.groupby('trade_date')['section1'].rank(method='min', pct=True)
        df_all['alpha_092'] = -np.where(df_all['section1'] > df_all['section2'], df_all['section1'], df_all['section2'])
        self.df = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        return self.df

    ## need nothing (good)
    def alpha098(self, df):
        ## 60日均值回归
        ####((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) /DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))###
        def cal_(df):
            section1 = ((df['close'].rolling(60).mean()).diff(60)) / df['close'].shift(60)
            df['alpha_098'] = np.where(section1 <= 0.05,
                                       (df['close'].rolling(60).min() - df['close']),
                                       -df['close'].diff(3))
            return df

        df = df.groupby('ts_code').apply(cal_)
        df[['alpha_098']] = df.groupby('trade_date')[['alpha_098']].rank(method='min', pct=True)
        self.df = df
        return self.df

    # execllent
    def alpha_099(self, df):  # 1766
        ## alpha098改造 5日价量横截面负相关
        ####(-1 * Rank(Cov(Rank(self.close), Rank(self.volume), 5)))###
        def cal_(df):
            # df['alpha_098'] = Cov(df['section1'], df['section2'], 6)
            df['alpha_099'] = Corr(df[['section1', 'section2']], 6)['corr']
            return df

        df['section1'] = df['revenue_ps'] / df['vwap']
        df['section2'] = np.log(df['volume'])
        #df[['section1', 'section2']] = df.groupby('trade_date')[['section1', 'section2']].rank(pct=True)
        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        df_all['alpha_099'] = df_all['section1'] * (-df_all['alpha_099'])
        self.df = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        return self.df

    ## need nothing (execllent)
    def alpha101(self, df):
        ## 长短线价量相关回归
        ###((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),RANK(VOLUME), 11))) * -1)
        def cal_(df):
            df['section2'] = -Corr(df[['section1', 'section2']], 11)['corr']
            section1 = (df['volume'].rolling(30).mean()).rolling(37).sum()
            section1 = pd.concat([df['section1'], section1], axis=1)
            df['section1'] = Corr(section1, 15)['corr']
            return df

        df['section1'] = df.groupby('ts_code')['log-ret'].cumsum()
        df[['section1', 'section2']] = df.groupby('trade_date')[['section1', 'volume']].rank(method='min', pct=True)

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        #df_all[['section1', 'section2']] = df_all.groupby('trade_date')[['section1', 'section2']].rank(method='min', pct=True)
        df_all['alpha_101'] = np.where(df_all['section1'] < df_all['section2'], df_all['section2']-df_all['section1'], df_all['section1']-df_all['section2'])
        self.df = df_all.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        return self.df

    ## need nothing
    def alpha102(self, df):
        ## 成交量增长买入，减少空仓
        ####SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100###
        def cal_(df):
            df['alpha_102'] = Sma(np.maximum(df['volume'].diff(2), 0), 14, 2) / Sma(abs(df['volume'].diff(2)), 14, 2)
            return df

        df = df.groupby('ts_code').apply(cal_)
        df['alpha_102'] = df.groupby('trade_date')['alpha_102'].rank(method='min', pct=True)
        self.df = df
        return self.df

    ## excellent
    def alpha103(self, df):
        ## 寻找低点持仓按天线性递减 横截面寻找更优
        ####((20-LOWDAY(LOW,20))/20)*100###
        def cal_(df):
            df['alpha_103'] = (20 - Lowday(df['low'], 20)) / 20
            # 改造
            df['alpha_103'] = df['alpha_103']/df['atr_6']
            return df

        df = df.groupby('ts_code').apply(cal_)
        #df['alpha_103'] = df.groupby('trade_date')['alpha_103'].rank(method='min', pct=True)
        self.df = df
        return self.df

    ## need nothing
    def alpha110(self, df):  # 1650
        ## 20日上涨动量
        ####SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100###
        def cal_(df):
            df['section1'] = np.maximum(df['high'] - df['close'].shift(1), 0)
            df['section1'] = df['section1'].rolling(20).sum()
            df['section2'] = np.maximum(df['close'].shift(1) - df['low'], 1e-2)
            df['section2'] = df['section2'].rolling(20).sum()
            df['alpha_110'] = df['section1'] / df['section2']
            return df

        df = df.groupby('ts_code').apply(cal_)
        df['alpha_110'] = df.groupby('trade_date')['alpha_110'].rank(method='min', pct=True)
        self.df = df.drop(columns=['section1', 'section2']).sort_values(by=['date'])
        return self.df

    # execllent
    def alpha111(self, df):  # 1789
        ## 动量回归
        ####SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)###
        def cal_(df):
            df['alpha_111'] = Sma(
                df['volume'] * (df['close'] - df['low'] - df['high'] + df['close']) / (df['high'] - df['low']), 11, 2) \
                              - Sma(
                df['volume'] * (df['close'] - df['low'] - df['high'] + df['close']) / (df['high'] - df['low']), 4, 2)
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        self.df[['alpha_111']] = self.df.groupby('trade_date')[['alpha_111']].rank(method='min', pct=True).apply(zscore)
        self.df['alpha_111'] = -1 / self.df['alpha_111']
        return self.df

    def alpha112(self, df):
        ##
        ####(SUM((CLOSE-DELAY(CLOSE,1)>0? CLOSE-DELAY(CLOSE,1):0),12) - SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12) + SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
        def cal_(df):
            section1 = pd.Series(np.where(df['close'].diff(1) > 0, df['close'].diff(1), 0), index=df.index)
            section2 = pd.Series(np.where(df['close'].diff(1) <= 0, abs(df['close'].diff(1)), 0), index=df.index)
            # print(section2)
            df['alpha_112'] = (section1.rolling(12).sum() - section2.rolling(12).sum()) / (
                section1.rolling(12).sum() + section2.rolling(12).sum())
            return df

        df = df.groupby('ts_code').apply(cal_)

        df[['alpha_112']] = df.groupby('trade_date')[['alpha_112']].rank(method='min', pct=True).apply(zscore)
        self.df = df
        return self.df

    def alpha113(self, df):
        ####(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2))))###
        def cal_(df):
            df['section1'] = df['log-ret'].cumsum().shift(5).rolling(20).mean()
            df['section2'] = Corr(df[['volume', 'close']], 2)['corr']
            df['section1'] = df['section1'] * df['section2']
            df['section2'] = Corr(pd.concat([df['close'].rolling(5).sum(), df['close'].rolling(20).sum()], axis=1), 2)['corr']
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique()):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)
        df_all[['section1', 'section2']] = df_all.groupby('trade_date')[['section1', 'section2']].rank(pct=True).apply(zscore)
        df_all['alpha_113'] = -df_all['section1'] * df_all['section2']
        self.df = df_all.drop(columns=['section1', 'section2'])
        return self.df

    # execllent
    def alpha116(self, df):
        ####REGBETA(CLOSE,SEQUENCE,20)###
        def cal_(df):
            df['alpha_116'] = -Regbeta(df['close'], Sequence(14) ** 2)
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        #df_all[['alpha_116']] = df_all.groupby('trade_date')[['alpha_116']].rank(method='min', pct=True)
        self.df = df_all
        return self.df

    # good
    def alpha119(self, df):
        ####(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))###
        def cal_(df):
            section1 = (df['volume'].rolling(5).mean()).rolling(26).sum()
            df['section1'] = Decaylinear(Corr(pd.concat([section1, df['vwap']], axis=1), 5)['corr'], 7)
            df['section2'] = df['volume'].rolling(15).mean()
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing1...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        df_all[['section1', 'section2', 'section3']] = df_all.groupby('trade_date')[
            ['section1', 'section2', 'open']].rank()

        def cal2_(df):
            df['section2'] = Decaylinear(Tsrank(Corr(df[['section2', 'section3']], 21)['corr'].rolling(9).min(), 7), 8)
            return df

        df = pd.DataFrame()
        for ts_code in tqdm(df_all.ts_code.unique(), desc='processing2...'):
            tmp = df_all.loc[df_all.ts_code == ts_code]
            tmp = cal2_(tmp)
            df = df.append(tmp)
        df['alpha_119'] = df['section1'] - df['section2']
        self.df = df.drop(columns=['section1', 'section2', 'section3'])

        return self.df

    # excellent
    def alpha122(self, df):
        ####(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)###
        def cal_(df):
            df['alpha_122'] = (Sma(Sma(Sma(np.log(df['close']), 13, 2), 13, 2), 13, 2) - (
                Sma(Sma(Sma(np.log(df['close']), 13, 2), 13, 2), 13, 2)).shift(1)) / \
                              (Sma(Sma(Sma(np.log(df['close']), 13, 2), 13, 2), 13, 2)).shift(1)
            df['alpha_122'] = -df['alpha_122']
            return df

        self.df = df.groupby('ts_code').apply(cal_).sort_values(by=['date'])
        return self.df

    def alpha128(self, df):
        def cal_(df):
            #### 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
            A = (df['low'] + df['close']) / 2
            section1 = pd.Series(np.where(A > A.shift(1), A * df['volume'], 0), index=df.index)
            section2 = pd.Series(np.where(A <= A.shift(1), A * df['volume'], 0), index=df.index)
            df['alpha_128'] = 100 - (100 / (1 + section1.rolling(14).sum() / section2.rolling(14).sum()))
            return df

        df = df.groupby('ts_code').apply(cal_)
        #df[['alpha_128']] = df.groupby('trade_date')[['alpha_128']].rank(method='min',pct=True)  # .apply(zscore)
        #df['alpha_128'] = (df['alpha_128'] - 0.5) * 2
        self.df = df
        return self.df

    def alpha130(self, df):  # 1657
        ####(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))###
        def cal_(df):
            section1 = pd.concat([(df['high'] + df['low']) / 2, df['volume'].rolling(40).mean()], axis=1)
            df['section1'] = Decaylinear(Corr(section1, 9)['corr'], 10)
            df['section2'] = Decaylinear(Corr(df[['vwap', 'volume']], 7)['corr'], 3)
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing1...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)
        df_all[['section1', 'section2']] = df_all.groupby('trade_date')[['section1', 'section2']].rank()
        df_all['alpha_130'] = df_all['section1'] / df_all['section2']
        #df_all['alpha_130'] = df_all.groupby('trade_date')['alpha_130'].rank(method='min', pct=True)
        self.df = df_all
        #self.df['alpha_130'] = (self.df['alpha_130'] - 0.5) * 2
        self.df = self.df.drop(columns=['section1', 'section2'])
        return self.df

    def alpha131(self, df):  # 1030
        ####(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))###
        def cal_(df):
            df['section1'] = df['vwap'].diff(1)
            section2 = pd.concat([df['close'], df['volume'].rolling(50).mean()], axis=1)
            df['section2'] = Tsrank(Corr(section2, 14)['corr'], 14)
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing1...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)
        df = df_all
        df['section1'] = df.groupby('trade_date')['section1'].rank(method='min', pct=True)
        df['alpha_131'] = df['section1'] ** df['section2']

        #df['alpha_131'] = df.groupby('trade_date')['alpha_131'].rank(method='min', pct=True)  # .apply(zscore)
        #df['alpha_131'] = (df['alpha_131'] - 0.5) * 2
        self. df = df.drop(columns=['section1', 'section2'])
        return self.df

    def alpha133(self, df):
        ####((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100###
        def cal_(df):
            df['alpha_133'] = (20 - Highday(df['high'], 20)) / 20 * 100 - ((20 - Lowday(df['low'], 20)) / 20) * 100
            return df

        df = df.groupby('ts_code').apply(cal_)
        df['alpha_133'] = df.groupby('trade_date')['alpha_133'].rank(method='min', pct=True)
        self.df = df
        return self.df

    ## excellent
    def alpha149(self, df):
        ## 越接近0.5 下跌可能性越大，需要全量测试
        ####REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),
        # FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)

        def cal_(df):
            cond = self.benchmark_return.loc[df.index].copy()
            # section = benchmark_return_acc.loc[df.index].copy()
            # df['section1'] = df['log-ret'].cumsum()
            df['section1'] = np.where(cond < 0, df['log-ret'], 0)
            df['section2'] = np.where(cond < 0, cond, 0)
            df['alpha_149'] = df['section2'].rolling(60).apply(
                lambda y: np.polyfit(df.loc[y.index]['section1'], y, deg=1)[0])
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        self.df = self.df.sort_values(by=['date'])
        #df['alpha_149'] = df.groupby('trade_date')['alpha_149'].rank(pct=True)
        return self.df

    def alpha172(self, df):
        ####MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
        def cal_(df):
            TR1 = df['high'] - df['low']
            TR2 = (df['high'] - df['close'].shift(1)).abs()
            TR3 = (df['low'] - df['close'].shift(1)).abs()
            TR = pd.Series(np.maximum(np.maximum(TR1, TR2), TR3), index=df.index)
            HD = df['high'] - df['high'].shift(1)
            LD = df['low'].shift(1) - df['low']
            cond1 = ((LD > 0) & (LD > HD))
            cond2 = ((HD > 0) & (HD > LD))
            part1 = pd.Series(np.where(cond1, LD, 0), index=df.index)
            part2 = pd.Series(np.where(cond2, HD, 0), index=df.index)
            df['alpha_172'] = ((part1.rolling(14).sum() - part2.rolling(14).sum()) / TR.rolling(14).sum()).abs()
            df['alpha_172'] = df['alpha_172'] / (
            (part1.rolling(14).sum() + part2.rolling(14).sum()) / TR.rolling(14).sum())
            df['alpha_172'] = -df['alpha_172'].rolling(6).mean()
            return df

        self.df = df.groupby('ts_code').apply(cal_).sort_values(by=['date'])
        return self.df

    ## good
    def alpha176(self, df):  # 1678
        ## 短线量价关系
        ####CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)###
        def cal_(df):
            df['section1'] = (df['close'] - df['low'].rolling(12).min()) \
                             / (df['high'].rolling(12).max() - df['low'].rolling(12).min())
            df['section2'] = pd.Series(np.log(df['volume']), index=df.index)
            return df

        df = df.groupby('ts_code').apply(cal_)
        df[['section1', 'section2']] = df.groupby('trade_date')[['section1', 'section2']].rank()

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp['alpha_176'] = -Corr(tmp[['section1', 'section2']], 6)['corr']
            df_all = df_all.append(tmp)

        self.df = df_all.drop(columns=['section1', 'section2'])
        return self.df


    def alpha190(self, df):

        def cal_(df):
            sub1 = df['close'] / df['close'].shift(1)
            sub2 = df['close'] / df['close'].shift(19)
            section1 = pd.Series(np.where(sub1 > (sub2 ** (1 / 20) - 1), 1, 0), index=df.index)
            section1 = section1.rolling(20).sum() - 1

            section2 = pd.Series(np.where(sub1 < (sub2 ** (1 / 20)), (sub1 - (sub2 ** (1 / 20) - 1)) ** 2, 0),
                                 index=df.index)
            section2 = section2.rolling(20).sum()

            section3 = section1 + 1
            section4 = pd.Series(np.where(sub1 > (sub2 ** (1 / 20)), (sub1 - (sub2 ** (1 / 20) - 1)) ** 2, 0),
                                 index=df.index)
            section4 = section4.rolling(20).sum()
            df['alpha_190'] = np.log(section1 * section2 / (section3 * section4))
            return df

        self.df = df.groupby('ts_code').apply(cal_)
        return self.df
