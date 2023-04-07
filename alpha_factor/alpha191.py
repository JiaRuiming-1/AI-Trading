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
        # fama_df.at[dt, 'trade_date'] = int(tmp.trade_date.unique()[0])
        fama_df.at[dt, 'SMB'] = SMB
        fama_df.at[dt, 'HML'] = HML
        fama_df.at[dt, 'MKT'] = tmp['log-ret'].mean() - benchmark_return[dt]

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


class Alpha191():
    def __init__(self, df):
        self.df = df
        self.benchmark_close = df.groupby('date').apply(benchmark_close_day)
        self.benchmark_open = df.groupby('date').apply(benchmark_open_day)
        self.benchmark_return = self.benchmark_close.pct_change().fillna(0.)
        self.fama_df = Getfama3factors(self.df, self.benchmark_return)

    ## need nothing (good)
    def alpha001(self, df):
        ## 成交量 与 涨幅关系
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

    ## need zscore and percent (test)
    def alpha002(self, df):
        ## K线形态变化
        ##### -1 * delta((((close-low)-(high-close))/(high-low)),1))####
        df['alpha_002'] = -(((df.close - df.low) - (df.high - df.close)) / (df.high - df.low)).diff(1)
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
    def alpha005(self, df):  # 1447
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

    ## need nothing
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

    # need percent
    def alpha010(self, df):
        ## 横截面相对价格突破
        ####(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))###
        def cal_(df):
            df['alpha_010'] = np.where(df['log-ret'] < 0, df['log-ret'].rolling(20).std(), df['close'])
            df['alpha_010'] = (df['alpha_010'] ** 2).rolling(window=5).max()
            return df

        df = df.groupby('ts_code').apply(cal_)
        df['alpha_010'] = df.groupby('trade_date')['alpha_010'].rank(method='min', pct=True)
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

        def rescale_(df):
            df['alpha_012'] = np.where(df['alpha_012'] > 0, df['alpha_012'] / df['alpha_012'].max(),
                                       df['alpha_012'] / df['alpha_012'].min())
            df['alpha_012'] = df['alpha_012'] * 2 - 1.
        df = df.groupby('ts_code').apply(rescale_)
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

    # need nothing
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
            df['alpha_016'] = -df['alpha_016'].rolling(5).max()
            df['alpha_016'] = df['alpha_016'] + 1.
            return df

        df_all['alpha_016'] = df_all.groupby('trade_date')['alpha_016'].rank(method='min', pct=True)
        df_all = df_all.groupby('ts_code').apply(cal_)
        self.df = df_all.drop(columns=['section1', 'section2'])
        return self.df

    # need nothing (good)
    def alpha017(self, df):
        ## 15日以来的最新上涨动力
        ####RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)###
        def cal1_(df):
            # df['alpha_017'] = df['vwap'] - df['vwap'].rolling(window=15).max()
            # ==== improve ====
            df['alpha_017'] = (df['vwap'] - df['vwap'].rolling(window=15).max()) / df['vwap']
            return df

        def cal2_(df):
            # df['alpha_017'] = df['alpha_017'] ** df['close'].diff(5).fillna(0.)
            # ====improve====
            df['alpha_017'] = df['alpha_017'] ** (df['close'].diff(5).fillna(0.) / df['close']) -1
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

    # need nothing (good)
    def alpha021(self, df):
        # 6日close均值~时间窗口回归
        ####REGBETA(MEAN(CLOSE,6),SEQUENCE(6))###
        def cal_(df):
            df['alpha_021'] = Regbeta(df['close'].rolling(6).mean(), Sequence(6))
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='alpha021 processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)

        self.df = df_all
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

        self.df = df.groupby('ts_code').apply(cal_)
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
        df['alpha_025'] = ((-df['section1']) * (1 + df['section3'])).fillna(-1.) + .8
        self.df = df.drop(columns=['section1', 'section2', 'section3'])
        return self.df

    # need percent
    def alpha026(self, df):
        ## 长线超跌买入
        ####((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))###
        def cal_(df):
            df['section1'] = (df['close'].rolling(7).mean() - df.close)
            df['section2'] = df['close'].shift(5)
            df['section2'] = Corr(df[['vwap', 'close']], 90)
            df['alpha_026'] = df['section1'] + df['section2']
            return df

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='alpha026 processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp = cal_(tmp)
            df_all = df_all.append(tmp)
        self.df = df_all.drop(columns=['section1', 'section2'])
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
    def alpha028(df):
        ## 3日夏普比率涨跌趋势
        ####3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)###
        def cal_(df):
            con1 = 3 * Sma((df['close'] - df['low'].rolling(9).min()) / (df['high'].rolling(9).max()
                                                                         - df['low'].rolling(9).min()), 3, 1)
            con2 = 2 * Sma(con1, 3, 1)
            df['alpha_028'] = con1 - con2
            return df

        df = df.groupby('ts_code').apply(cal_)
        return df

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
        ## 三因子beta回归 >0.5 超跌买入
        ####WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML， 60))^2,20)###
        def cal_(df):
            obj = df.rolling(window=60, method='table')
            s = []
            for o in obj:
                s.append(RegResi(o, self.fama_df) ** 2)
            return pd.DataFrame(s, index=df.index, columns=['beta'])

        df_all = pd.DataFrame()
        for ts_code in tqdm(df.ts_code.unique(), desc='alpha030 processing...'):
            tmp = df.loc[df.ts_code == ts_code]
            tmp['alpha_030'] = cal_(tmp)['beta']
            df_all = df_all.append(tmp)
        self.df = df_all
        return self.df


if __name__ == '__main__':
    # df = pd.read_csv('tushare_data/raw_20180103_20230327.csv').iloc[:, 1:]
    df = pd.read_csv('tmp.csv').iloc[:, 1:]
    df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.set_index(['date']).sort_values(by=['date'])
    df['volume'] = df['amount'] / df['close']
    df['log-ret'] = df.groupby('ts_code')['close'].pct_change()
    df['vwap'] = (df['low'] + df['high']) / 2
    # PICK TOP 5 TICKERS
    ts_code_list = np.append(df.ts_code.unique()[:9], '603538.SH')
    print(ts_code_list)
    universe = df.loc[df.ts_code.isin(ts_code_list)].copy(deep=True)
