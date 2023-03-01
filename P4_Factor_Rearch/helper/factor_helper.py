import pandas as pd
import numpy as np
from tqdm import tqdm
import stockstats

class AverageByWindow(pd.DataFrame):
    """
        average dollar volume in indicate time widow

        Parameters
        ----------
        data : DateFrame
        win_lenth: averge time window lenth
        use_columns:[str] what columns will be calculated in window
    """
    def __init__(self,data, use_columns = [], window_length = 1):
        super(AverageByWindow, self).__init__(data)
        self.window_lenth = window_length
        if len(use_columns) == 0:
            self.use_column = self.columns
        else:
            self.use_column = use_columns

        self.df = self[self.use_column].rolling(window = window_length).mean()
        new_cols = self.columns.values.copy()
        for col in self.use_column:
            new_cols = np.append(new_cols,[col+'_'+str(window_length)+'average'])
        self.df = pd.concat([self, self.df],axis=1)
        self.df.columns = new_cols
        #self = self.merge(self.df[["ts_code", "date", indicator]], on=["ts_code", "date"], how="left")

    def add_indicators(self):
        return self.df

    def top(self, num, ticker_column, value_column):
        '''
        pick up top number of tickers which indicated by column sum value

        :param num: (integer) how many tickers return
        :param ticker_column: (str) which column name to get tickers name
        :param value_column: (str) column name to rank
        :return: top number tickers dataframe
        '''
        self.df = self.df[self.window_lenth:]
        df = self.df.pivot(index='date', columns=ticker_column, values=value_column).fillna(0)
        df = df.sum(axis=0).sort_values(ascending=False)[:num]
        stocks_name = df.index.values
        return self.df.loc[self.df[ticker_column].isin(stocks_name)]

class IndicatorHelper(pd.DataFrame):
    def __init__(self, data):
        super(IndicatorHelper, self).__init__(data)

        self.stocks = stockstats.StockDataFrame.retype(self.copy())

    def add_technical_indicator(self, tech_indeicator_list, unique_ticker):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param ticker: (df) pandas dataframe
        :param tech_indeicator_list list
        :return: (df) pandas dataframe
        """
        df = self.stocks.sort_values(by=["ts_code", "date"])

    def add_by_basetable(self, ticker_column, base_table, add_columns):
        '''
        add base indicator from base_table by tushare. example industry, pe etc.
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
        df_new = pd.DataFrame()
        for ts_code in tqdm(self[ticker_column].unique(), desc='ticker/tickers'):
            tmp = self.loc[self[ticker_column] == ts_code]
            tmp[add_columns] = base_table.loc[base_table[ticker_column] == ts_code][add_columns].values[0]
            df_new = df_new.append(tmp)
        return df_new


