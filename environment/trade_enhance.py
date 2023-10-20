import os

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
from math import floor
from typing import Optional, Tuple, Union
import talib


class InfoContainer(object):
    def __init__(self):
        super(InfoContainer, self).__init__()
        self.offset = 0  # 資料定位
        self.balance = 1_000_000  # 現金餘額
        self.net = 0  # 淨損益
        # self.net_exclude_settlement = 0  # 淨損益 (不含結算)
        self.cost = 0  # 平均成本
        self.cost_total = 0  # 總成本
        self.hold = 0  # 持有量
        self.holding_count = 0  # 不動作次數

    def reset(self):
        self.balance = 1_000_000
        self.cost = 0
        self.hold = 0


class DataLocater(object):
    def __init__(self, index_path: Union[str, bytes, os.PathLike], data_root: Union[str, bytes, os.PathLike],
                 random_state: Optional[int] = None):
        super(DataLocater, self).__init__()
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

    def get_index(self) -> str:
        return self.index[self.offset - 1]

    def next(self) -> pd.DataFrame:
        data = pd.read_csv(f'{self.data_root}/個股/{self.index[self.offset]}.csv').reset_index(drop=True).set_index('Date')

        ii = pd.read_csv(f'{self.data_root}/法人買賣超日報_個股/{self.index[self.offset]}.csv', index_col=0)
        ii = ii.reset_index(drop=True).set_index('Date')

        ii = ii.apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        ii = ii[['外陸資買賣超股數(不含外資自營商)', '外資自營商買賣超股數', '投信買賣超股數', '自營商買賣超股數',
                 '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)', '三大法人買賣超股數']]
        ii = ii.apply(lambda x: x.apply(lambda y: 1 if float(y) > 0 else -1 if float(y) < 0 else 0))
        ii = ii.shift(1)  # 往下偏移一天 (因為當天結束才會統計買賣超資訊，實務上交易日當天是不知道當天買賣超資訊的)
        ii = ii.fillna(0)  # 用 0 補空值 (影響應該不會太大)

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        data = pd.concat([data, ii], axis=1).dropna()

        # 合併完，再計算技術指標 (避免 dropna 把技術指標開頭資料刪除)
        # SMA
        data['MA5'] = talib.MA(data['Close'], timeperiod=5)
        data['MA10'] = talib.MA(data['Close'], timeperiod=10)
        data['MA20'] = talib.MA(data['Close'], timeperiod=20)
        data['MA60'] = talib.MA(data['Close'], timeperiod=60)
        # MACD
        data['MACD'], data['MACDsignal'], data['MACDhist'] = \
            talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        # RSI
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        # CCI
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
        # ADX
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

        data = data[['Close',  # 收盤價
                     '外陸資買賣超股數(不含外資自營商)', '外資自營商買賣超股數', '投信買賣超股數', '自營商買賣超股數',
                     '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)', '三大法人買賣超股數',  # 法人買賣超
                     'MA5', 'MA10', 'MA20', 'MA60',  # SMA
                     'MACD', 'MACDsignal', 'MACDhist',  # MACD
                     'RSI',  # RSI
                     'CCI',  # CCI
                     'ADX'  # ADX
                     ]]

        data = data.fillna(0)  # 補空值

        if self.offset < len(self.index) - 1:
            self.offset += 1
        else:
            self._set_index(random_state=self.random_state)
            self.offset = 0

        return data


class Env(gym.Env):
    def __init__(self, observation_space: spaces.Box):
        """

        """
        super(Env, self).__init__()

        self.observation_space = observation_space  # 直接用傳入的資料取樣
        self.action_space = spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)

        '''
        資訊類 Container
        '''
        self.data_getter = DataLocater(index_path='data/ind.csv', data_root='data/train', random_state=25)
        self.data = self.data_getter.next()
        self.info = InfoContainer()

        '''
        正規化器
        '''
        # self.scaler = None  # set in reset()

    def _decode_action(self, action: np.float32) -> Tuple[Optional[bool], int]:
        # action 是比例
        if action > 0:  # Buy
            project = self.info.balance * action  # 愈買入金額
            trade = project / self.data.iloc[self.info.offset]['Close']  # 實際買入股數
            trade = floor(trade)  # 無條件捨去 (最小單位為1股，且不可買超過餘額)
            return True, trade
        elif action < 0:  # Sell
            project = self.info.hold * action  # 愈賣出股數
            trade = project  # 實際賣出股數
            trade = -floor(trade)  # 無條件捨去 (最小單位為1股，且不可賣超過持有量)
            return False, trade
        else:  # Hold
            return None, 0

    def _locate_data(self, offset: int) -> pd.DataFrame:
        data = self.data.iloc[offset].copy()  # .drop(['stock_num'], axis=1)
        data['balance'] = self.info.balance  # 把餘額加入
        data['hold'] = self.info.hold  # 把持有量加入
        return data

    def reset(self, seed=None, *args, **kwargs) -> tuple[np.ndarray, dict]:
        # 理論上，這裡的 observation 會與重置前的 observation 相同 (offset 相同)
        print(f'reset: {self.data_getter.get_index()}')

        self.info.reset()  # 重置部分資訊

        observation = self._locate_data(self.info.offset)
        info = {
            'stock_num': self.data_getter.get_index(),
        }

        return observation.to_numpy().astype(np.float32), info

    def _calculate_trade(self, buy_or_sell: bool, trade: int) -> int:
        if buy_or_sell is None:  # Hold
            self.info.holding_count += 1

            if self.info.holding_count >= 10:  # 連續第 10 天沒動作
                # return -3  # 懲罰
                pass  # 有可能影響訓練，先不懲罰做觀察

            return 0

        self.info.holding_count = 0  # 重置不動作次數

        if trade == 0:  # 沒有交易
            return -1  # 懲罰

        # Buy or Sell
        if buy_or_sell:  # Buy
            price = self._locate_data(self.info.offset)['Close']  # 買入單價
            self.info.cost = (self.info.cost * self.info.hold + price * trade) / (self.info.hold + trade)  # 計算加權平均成本
            if np.isnan(self.info.cost):
                print(f'{self.info.cost} * {self.info.hold} + {price} * {trade} / ({self.info.hold} + {trade})')
                raise ValueError('cost is nan')
            self.info.balance -= price * trade  # 扣除買入金額
            self.info.cost_total += price * trade  # 累計成本
            self.info.hold += trade  # 持有量增加
            return 0
        else:  # Sell
            price = self._locate_data(self.info.offset)['Close']  # 賣出單價
            # self.info.cost  # 加權平均成本不會改變
            self.info.balance += price * trade  # 加回賣出金額
            self.info.net += (price - self.info.cost) * trade  # 計算損益
            self.info.hold -= trade  # 持有量減少
            if np.isnan(self.info.net):
                print(f'{self.info.cost}')
                raise ValueError('net is nan')
            if price > self.info.cost:
                return 5  # 有賺錢所以獎勵
            else:
                return 0  # 沒賺錢，不給獎勵

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        buy_or_sell, trade = self._decode_action(action)

        reward = self._calculate_trade(buy_or_sell, trade)
        terminated = True if self.info.offset == len(self.data) - 1 else False
        truncated = False  # 這表示因超時導致的終止，我們不會有這種情況

        if terminated:  # 已經結束這檔股票的交易，進行結算
            if self.info.hold > 0:
                price = self._locate_data(self.info.offset)['Close']  # 賣出單價
                self.info.net += price * self.info.hold  # 計算損益
                if price > self.info.cost and reward == 0:  # 如果最終持有價格大於平均成本，且本次沒有獲得獎勵，給予較低獎勵
                    reward = 3  # 有賺錢所以獎勵

            self.data = self.data_getter.next()  # 獲取資料
            self.info.offset = 0  # 資料定位歸零
        else:
            self.info.offset += 1  # 下移資料定位 (換日)

        observation = self._locate_data(self.info.offset).to_numpy().astype(np.float32)
        info = {
            'cost': self.info.cost_total,
            'hold': self.info.hold,
            'holding_count': self.info.holding_count,
            'net': self.info.net,
            # 'net_exclude_settlement': self.info.net_exclude_settlement,
            'finish': False  # TODO: 晚點改
        }

        return observation, reward, terminated, truncated, info


class TensorboardCallback(BaseCallback):
    """
    Tensorboard 記錄
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('env/balance', self.training_env.get_attr('info')[0].balance)
        self.logger.record('env/hold', self.training_env.get_attr('info')[0].hold)
        self.logger.record('env/holding_count', self.training_env.get_attr('info')[0].holding_count)
        self.logger.record('env/net', self.training_env.get_attr('info')[0].net)
        return True


if __name__ == '__main__':
    pass
