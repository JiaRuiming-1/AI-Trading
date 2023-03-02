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
        universe = universe.dropna()
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
    def __init__(self,data):
        super(WinnerAndLoser, self).__init__(data)
        self.df = self

    def _regression(self, data):
        regression = sfa.ols(formula='pct_chg ~ t_dir + t_velocity', data=data)
        model = regression.fit()
        data['win_lose'] = model.params.t_dir * abs(model.params.t_velocity)
        print('\r processing factors step/total {}/{}'.format(self.count, self.shape[0]), end='')
        self.count += 1
        return  data


    def calculate(self):
        '''
        convert time to value
        regress return to get mu and beta each time
        add facotor mu*beta to colomns
        :return: dataframe
        '''
        self.count = 1
        tmp_df = self.copy()[['date', 'ts_code', 'pct_chg']]
        tmp_df['t_dir'] = (self.date - pd.Timestamp("1990-01-01")) / (pd.Timedelta('1d') * 1000)
        tmp_df['t_velocity'] = tmp_df['t_dir'] ** 2
        tmp_df = tmp_df.apply(lambda x: self._regression(x), axis=1)
        self.df['win_lose'] = tmp_df['win_lose']
        return  self

    def get_factor(self):
        return self.df


class SkewandMomentum(pd.DataFrame):
    """
        Expected Skewness and Momentum Factor Constructor

        Parameters
        ----------
        data : DateFrame
    """
    def __init__(self,data):
        super(SkewandMomentum, self).__init__(data)
        self.df = self

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
            roll_obj = stock.rolling(20)['pct_chg']
            stock['skew_momentum'] = roll_obj.skew() * roll_obj.median(axis=0)
            stock.fillna(method='backfill', inplace=True)
            tmp_df = tmp_df.append(stock)
        self.df = self.df.merge(tmp_df[["ts_code", "date", "skew_momentum"]], on=["ts_code", "date"], how="left")
        self.df = self.df.sort_values(by=["date", "ts_code"])
        return self

    def get_factor(self):
        return self.df


class BollingerAndResidual(pd.DataFrame):
    """
        Custom Factor Constructor

        Parameters
        ----------
        data : DateFrame
        residuals : DateFrame
    """
    def __init__(self,data, residuals):
        super(BollingerAndResidual, self).__init__(data)
        self.df = self
        self.residuals = residuals

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
            tmp_df = self.df.loc[self.df.ts_code == ticker][['ts_code', 'date', 'boll_ub','boll_lb','close']]
            residual = self.residuals.loc[self.residuals.index==ticker].values[0]
            tmp_df['custom_factor'] = (tmp_df['boll_ub'] + tmp_df['boll_lb'] - 2 * tmp_df['close']) \
                                      * residual / 1000
            factor_df = factor_df.append(tmp_df)

        self.df = self.df.merge(factor_df[["ts_code", "date", "custom_factor"]], on=["ts_code", "date"], how="left")
        self.df = self.df.sort_values(by=["date", "ts_code"])
        return self

    def get_factor(self):
        return self.df
