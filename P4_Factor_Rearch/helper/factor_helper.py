import pandas as pd
import numpy as np

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

if __name__ == '__main__':
    df = pd.read_csv('eod-quotemedia.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date')
    df = AverageByWindow(df.drop_duplicates(), use_columns=['adj_volume'], window_length=120)
    pool = df.top(50, 'ticker', 'adj_volume_120average')
    print(pool)
