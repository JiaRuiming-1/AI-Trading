import pandas as pd
import numpy as np
from tqdm import tqdm
import stockstats
import statsmodels.formula.api as sfa

import warnings
warnings.filterwarnings('ignore')

class IndicatorHelper(pd.DataFrame):
    """
        add indicators to dataframe

        Parameters
        ----------
        data : DateFrame
    """

    def __init__(self, data):
        super(IndicatorHelper, self).__init__(self._process_data(data))

        self.stocks = stockstats.StockDataFrame.retype(self.copy())
        self.df = self

    def add_technical_indicator(self, tech_indicator_list):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param ticker: (df) pandas dataframe
        :param tech_indeicator_list list
        :return: (df) pandas dataframe
        """
        self.df = self.df.sort_values(by=["ts_code", "date"])
        unique_ticker = self.df.ts_code.unique()

        indicator_df = pd.DataFrame()
        for i in tqdm(range(len(unique_ticker)), desc='add tech indicators'):
            temp_indicator = self.stocks[self.stocks.ts_code == unique_ticker[i]][tech_indicator_list]
            temp_indicator = pd.DataFrame(temp_indicator)
            temp_indicator["ts_code"] = unique_ticker[i]
            temp_indicator["date"] = self.df[self.df.ts_code == unique_ticker[i]]["date"].to_list()
            temp_indicator.fillna(method='backfill', inplace=True)
            indicator_df = indicator_df.append(temp_indicator, ignore_index=True)

        self.df = self.df.merge(indicator_df[["ts_code", "date"]+ tech_indicator_list], on=["ts_code", "date"], how="left")
        self.df = self.df.sort_values(by=["date", "ts_code"])
        return self.df

    def add_by_basetable(self, ticker_column, base_table, add_columns):
        '''
        add base common indicator from base_table. example industry, area. name etc.
        base table like:
        =======================
        ticker industry  pe
        A      economic  10.1
        B      food      22,2
        B      service   33.3
        =======================
        :param (str) ticker_column: the name of ticker column name
        :param (dataframe) base_table: basic message table
        :param (list) base_table: add what attributes to new dataframe eg: ['industry', 'pe']
        :return:(dataframe) dataframe with added indicator
        '''
        df = pd.DataFrame()
        for ts_code in tqdm(self.df[ticker_column].unique(), desc='add fundamental info'):
            tmp = self.df.loc[self.df[ticker_column] == ts_code]
            tmp[add_columns] = base_table.loc[base_table[ticker_column] == ts_code][add_columns].values[0]
            df = df.append(tmp)

        self.df = self.df.merge(df[["ts_code", "date"] + add_columns], on=["ts_code", "date"],how="left")
        self.df = self.df.sort_values(by=["date", "ts_code"])
        return self.df

    def top(self, num, index, ticker_column, value_column):
        '''
        pick up top number of tickers which indicated by column sum value

        :param num: (integer) how many tickers return
        :param ticker_column: (str) which column name to get tickers name
        :param value_column: (str) column name to rank
        :return: top number tickers dataframe
        '''
        df = self.df.pivot(index=index, columns=ticker_column, values=value_column).fillna(0)
        df = df.sum(axis=0).sort_values(ascending=False)[:num]
        stocks_name = df.index.values
        self.df = self.df.loc[self.df[ticker_column].isin(stocks_name)]
        # self.stocks = stockstats.StockDataFrame.retype(self.df.copy())
        return self.df

    def _process_data(self, data):
        '''
        process date as date time type and order by time
        :return: processed data
        '''
        universe = data.sort_index(axis=0, ascending=False)
        # convert date to standard string format, easy to filter
        universe["date"] = pd.to_datetime(universe["trade_date"], format='%Y%m%d')
        universe["date"] = universe.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        universe = universe.sort_values(by=["date", "ts_code"]).reset_index(drop=True)
        universe['date'] = pd.to_datetime(universe['date'])
        return universe

class CloseToOpen(pd.DataFrame):
    """
        Overnight Return Factor Constructor

        Parameters
        ----------
        data : DateFrame
    """
    def __init__(self,data):
        super(CloseToOpen, self).__init__(data)
        self.df = self

    def calculate(self):
        '''
        add open-close as a column named close_to_return
        :return: dataframe
        '''
        tmp_df = pd.DataFrame()
        for stock_tuple in tqdm(self.groupby('ts_code'), desc='close_to_open'):
            stock = stock_tuple[1]
            stock['close_to_open'] = (stock['open'].shift(-1).fillna(method='backfill') - stock['close'])/stock['close']
            stock.fillna(method='ffill', inplace=True)
            tmp_df = tmp_df.append(stock)
        self.df = self.df.merge(tmp_df[["ts_code", "date", "close_to_open"]], on=["ts_code", "date"], how="left")
        return self

    def get_factors(self):
        '''
        calculate close_to_open_5_sma, close_to_open_25_sma by Indicator Helper class
        :return: Dateframe
        '''
        ind_helper = IndicatorHelper(self.df)
        self.df = ind_helper.add_technical_indicator(['close_to_open_5_sma', 'close_to_open_25_sma'])
        self.df['close_to_open_25_sma'] = - self.df['close_to_open_25_sma']
        return self.df


class WinnerAndLoser(pd.DataFrame):
    """
        Winner and Loser Factor Constructor

        Parameters
        ----------
        data : DateFrame
    """
    def __init__(self,data, win_length=60):
        super(WinnerAndLoser, self).__init__(data)
        self.df = self
        self.win_lenth = win_length

    def _regression(self, data):
        df = pd.DataFrame(data, columns=['pct_chg'])
        df['t_dir'] = np.arange(self.win_lenth)+1
        df['t_velocity'] = df['t_dir'] ** 2
        regression = sfa.ols(formula='pct_chg ~ t_dir + t_velocity', data=df)
        model = regression.fit()
        data['win_lose'] = model.params.t_dir * abs(model.params.t_velocity)
        return  data['win_lose']


    def calculate(self):
        '''
        convert time to value
        regress return to get mu and beta each time
        add facotor mu*beta to colomns
        :return: dataframe
        '''
        tickers = self.df.ts_code.unique()
        factor_df = pd.DataFrame()
        for ticker in tqdm(tickers, desc='win and lose'):
            tmp_df = self.df.loc[self.df.ts_code == ticker][['date', 'ts_code', 'pct_chg']]
            tmp_df['win_lose'] = tmp_df['pct_chg'].rolling(self.win_lenth).apply(self._regression)
            factor_df = factor_df.append(tmp_df)
        self.df = self.df.merge(factor_df[["ts_code", "date", "win_lose"]], on=["ts_code", "date"], how="left")
        return  self

    def get_factor(self):
        return self.df


class SkewandMomentum(pd.DataFrame):
    """
        Expected Skewness and Momentum Factor Constructor

        Parameters
        ----------
        data : DateFrame
        win_lenth: int
            the rolling window length of days
    """
    def __init__(self,data, win_length):
        super(SkewandMomentum, self).__init__(data)
        self.df = self
        self.win_length = win_length

    def calculate(self):
        '''
        convert time to value
        regress return to get mu and beta each time
        add facotor mu*beta to colomns
        :return: dataframe
        '''
        tmp_df = pd.DataFrame()
        for stock_tuple in tqdm(self.groupby('ts_code'), desc='skew and momentum'):
            stock = stock_tuple[1]
            roll_obj = stock.rolling(self.win_length)['pct_chg']
            stock['skew_momentum'] = roll_obj.skew() * roll_obj.median(axis=0)
            tmp_df = tmp_df.append(stock)
        self.df = self.df.merge(tmp_df[["ts_code", "date", "skew_momentum"]], on=["ts_code", "date"], how="left")
        self.df = self.df.sort_values(by=["date", "ts_code"])
        return self

    def get_factor(self):
        return self.df

class SuperTrend(pd.DataFrame):
    """
        Custom Factor Constructor
        Parameters
        ----------
        data : DateFrame
    """
    def __init__(self,data):
        super(SuperTrend, self).__init__(data)
        self.df = self

    def calculate(self):
        '''
        factor = (boll_ub + boll_lb - 2 * close) * residuals / 1000
        add facotor to colomns
        :return: dataframe
        '''
        self.df = self.df.sort_values(by=["ts_code", "date"])
        unique_ticker = self.df.ts_code.unique()

        factor_df = pd.DataFrame()
        for ticker in tqdm(unique_ticker, desc='custom factor'):
            tmp_df = self.df.loc[self.df.ts_code == ticker][['ts_code', 'date', 'close', 'supertrend_ub', 'supertrend_lb']]
            up = tmp_df['close'] - tmp_df['supertrend_ub']
            down =  tmp_df['close'] - tmp_df['supertrend_lb']
            tmp_df['supertrend_factor'] = np.where(up > 0,up, np.where(down < 0 ,down, 0))
            factor_df = factor_df.append(tmp_df)

        self.df = self.df.merge(factor_df[["ts_code", "date", "supertrend_factor"]], on=["ts_code", "date"], how="left")
        self.df = self.df.sort_values(by=["date", "ts_code"])
        return self

    def get_factor(self):
        return self.df

class FatherFactor(pd.DataFrame):
    """
        The factor created by my father

        Parameters
        ----------
        data : DateFrame
        win_lenth: int
            the rolling window length of days
    """
    def __init__(self,data, win_length):
        super(FatherFactor, self).__init__(data)
        self.df = self
        self.win_length = win_length

    def calculate(self):
        '''
        abs(market cap - 2.5 billion) close to 0
        abs((max price - now price)/ max price) - 70%) close to 0
        abs((now price - begin price)) close to 0
        abs(std(amount_120_sma) - 2%) close to 0

        :return: dataframe
        '''

        return self

    def get_factor(self):
        return self.df


if __name__ == '__main__':
    universe_raw = pd.read_csv('../20180101-20210101.csv').iloc[:, 1:]
    tickers = universe_raw['ts_code'].unique()[:4]
    universe = universe_raw.loc[universe_raw.ts_code.isin(tickers)]

    universe = IndicatorHelper(universe)

    wl = WinnerAndLoser(universe, win_length=120).calculate()
    universe = wl.get_factor()

    print(universe)
