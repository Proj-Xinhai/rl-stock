from typing import Union, Optional
import os
import pandas as pd
from sklearn.utils import shuffle
from datetime import datetime
import yfinance as yf


class BasicDataLocator(object):
    def __init__(self,
                 index_path: Union[str, bytes, os.PathLike],
                 data_root: Union[str, bytes, os.PathLike],
                 start: str,
                 end: str,
                 online: bool = False,
                 stock_id: Optional[list] = None,
                 random_state: Optional[int] = None):
        super(BasicDataLocator, self).__init__()
        self.random_state = random_state
        self.data_root = data_root
        self.index_path = index_path
        self.index = None

        self.start = datetime.fromisoformat(start).astimezone()
        self.end = datetime.fromisoformat(end).astimezone()
        self.online = online
        self.stock_id = stock_id
        if self.online and self.stock_id is None and len(self.stock_id) != 1:
            raise ValueError('online mode just for backtest, must specify stock_id, and the length of stock_id must be 1')

        self._set_index(random_state=self.random_state)

        self.offset = 0  # 資料定位

    def _set_index(self, random_state: Optional[int] = None):
        if self.online:
            self.index = self.stock_id
        else:
            ind = pd.read_csv(self.index_path)
            ind = ind['代號'].to_list()
            ind = shuffle(ind, random_state=random_state)
            self.index = ind

    def _get_current_data(self) -> Optional[pd.DataFrame]:
        if self.online:
            api = yf.Ticker(f'{self.stock_id[0]}.TW')
            start = self.start
            end = self.end + pd.Timedelta(days=1)  # end must add one day to get the last day of the data
            data = api.history(start=self.start, end=end, auto_adjust=False)
            if data.empty:
                data = None
        else:
            data = pd.read_csv(f'{self.data_root}/個股/{self.index[self.offset]}.csv')
            data = data[data['Date'] >= str(self.start)]
            data = data[data['Date'] <= str(self.end)]
            data = data.reset_index(drop=True).set_index('Date')
        return data

    def get_index(self) -> str:
        return self.index[self.offset - 1]

    def next(self) -> pd.DataFrame:
        raise NotImplementedError('next not implemented')


if __name__ == '__main__':
    raise NotImplementedError('this file not meant to be executed')
