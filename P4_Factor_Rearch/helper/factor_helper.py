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

    def add_average_indicators(self):
        '''
        add sma to indicate columns
        :return:
        '''
        self.df = self[self.use_column].rolling(window=self.window_length).mean()
        new_cols = self.columns.values.copy()
        for col in self.use_column:
            new_cols = np.append(new_cols, [col + '_' + str(self.window_length) + 'sma'])
        self.df = pd.concat([self, self.df], axis=1)
        self.df.columns = new_cols
        return self.df

    def top(self, num, index, ticker_column, value_column):
        '''
        pick up top number of tickers which indicated by column sum value

        :param num: (integer) how many tickers return
        :param ticker_column: (str) which column name to get tickers name
        :param value_column: (str) column name to rank
        :return: top number tickers dataframe
        '''
        df = self.pivot(index=index, columns=ticker_column, values=value_column).fillna(0)
        df = df.sum(axis=0).sort_values(ascending=False)[:num]
        stocks_name = df.index.values
        return self.loc[self[ticker_column].isin(stocks_name)]


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
        self.df = self.copy()

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
        for ts_code in tqdm(self[ticker_column].unique(), desc='add fundamental info'):
            tmp = self.df.loc[self.df[ticker_column] == ts_code]
            tmp[add_columns] = base_table.loc[base_table[ticker_column] == ts_code][add_columns].values[0]
            df = df.append(tmp)

        self.df = self.df.merge(df[["ts_code", "date"] + add_columns], on=["ts_code", "date"],how="left")
        self.df = self.df.sort_values(by=["date", "ts_code"])
        return self.df

    def _process_data(self, data):
        '''
        process date as date time type and order by time
        :return:
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

if __name__ == '__main__':
    # load data from csv
    universe = pd.read_csv('../20180101-20210101.csv').iloc[:, 1:]
    fundamental = pd.read_csv('../fundamental_20180101.csv').iloc[:, 1:]

    universe = AverageByWindow(universe)
    universe = universe.top(500, index='trade_date', ticker_column='ts_code', value_column='ma_v_120')

    universe = IndicatorHelper(universe)
    # tech_indicator_list = ['boll_ub','boll_lb']
    # universe = universe.add_technical_indicator(tech_indicator_list)
    universe = universe.add_by_basetable('ts_code', fundamental, ['industry', 'name'])
    pd.read_csv()
    print(universe)