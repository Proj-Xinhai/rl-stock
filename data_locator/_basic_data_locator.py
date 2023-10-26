from typing import Union, Optional
import os
import pandas as pd
from sklearn.utils import shuffle


class BasicDataLocator(object):
    def __init__(self, index_path: Union[str, bytes, os.PathLike], data_root: Union[str, bytes, os.PathLike],
                 random_state: Optional[int] = None):
        super(BasicDataLocator, self).__init__()
        self.random_state = random_state
        self.data_root = data_root
        self.index_path = index_path
        self.index = None
        self._set_index(random_state=self.random_state)

        self.offset = 0  # 資料定位

    def _set_index(self, random_state: Optional[int] = None):
        ind = pd.read_csv(self.index_path)
        ind = ind['代號'].to_list()
        ind = shuffle(ind, random_state=random_state)
        self.index = ind

    def _get_current_data(self) -> pd.DataFrame:
        data = pd.read_csv(f'{self.data_root}/個股/{self.index[self.offset]}.csv')
        return data.reset_index(drop=True).set_index('Date')

    def get_index(self) -> str:
        return self.index[self.offset - 1]

    def next(self) -> pd.DataFrame:
        raise NotImplementedError('next not implemented')


if __name__ == '__main__':
    raise NotImplementedError('this file not meant to be executed')
