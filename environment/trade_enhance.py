import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable, Any
from math import floor
from typing import Optional, Tuple


class InfoContainer(object):
    def __init__(self):
        super(InfoContainer, self).__init__()
        self.offset = 0  # 資料定位
        self.balance = 1_000_000  # 現金餘額
        self.net = 0  # 淨損益
        self.net_exclude_settlement = 0  # 淨損益 (不含結算)
        self.cost = 0  # 平均成本
        self.cost_total = 0  # 總成本
        self.hold = 0  # 持有量
        self.holding_count = 0  # 不動作次數

    def reset(self):
        self.balance = 1_000_000
        self.cost = 0
        self.hold = 0


class Env(gym.Env):
    def __init__(self, data: pd.DataFrame):
        """

        """
        super(Env, self).__init__()

        self.observation_space = None  # 直接用傳入的資料取樣
        self.action_space = spaces.Box(low=-np.ones(1), high=np.ones(1), shape=(1, 1), dtype=np.float32)

        '''
        資訊類 Container
        '''
        self.data = data
        self.info = InfoContainer()

        '''
        正規化器
        '''
        self.scaler = None  # set in reset()

    def _decode_action(self, action: np.float32) -> Tuple[Optional[bool], int]:
        # action 是比例
        if action > 0:  # Buy
            project = self.info.balance * action  # 愈買入金額
            trade = project / self.data.iloc[self.info.offset]['Close'].value[0]  # 實際買入股數
            trade = floor(trade)  # 無條件捨去 (最小單位為1股，且不可買超過餘額)
            return True, trade
        elif action < 0:  # Sell
            project = self.info.hold * action  # 愈賣出股數
            trade = project  # 實際賣出股數
            trade = floor(trade)  # 無條件捨去 (最小單位為1股，且不可賣超過持有量)
            return False, trade
        else:  # Hold
            return None, 0

    def _locate_data(self, offset: int) -> np.ndarray:
        data = self.data.iloc[offset].drop(['stock_id'], axis=1)
        data['hold'] = self.info.hold  # 把持有量加入
        return data.to_numpy().astype(np.float32)

    def _get_stock_num(self, offset: int) -> str:
        return self.data.iloc[offset]['stock_id'].value[0]

    def reset(self, seed=None, *args, **kwargs) -> tuple[np.ndarray, dict]:
        # 理論上，這裡的 observation 會與重置前的 observation 相同 (offset 相同)
        print(f'reset: {self._get_stock_num(self.info.offset)}')

        self.info.reset()  # 重置部分資訊

        observation = self._locate_data(self.info.offset)
        info = {
            'stock_num': self._get_stock_num(self.info.offset),
        }

        return observation, info

    def _calculate_trade(self):
        pass

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        buy_or_sell, trade = self._decode_action(action)

        reward = 0

        self.info.offset += 1  # 下移資料定位 (換日，或目前股票最後一天，換股)
        observation = self._locate_data(self.info.offset)
        # reward
        terminated = \
            True if self._get_stock_num(self.info.offset) != self._get_stock_num(self.info.offset - 1) else False
        truncated = False  # 這表示因超時導致的終止，我們不會有這種情況
        info = {
            'cost': self.info.cost_total,
            'hold': self.info.hold,
            'holding_count': self.info.holding_count,
            'net': self.info.net,
            'net_exclude_settlement': self.info.net_exclude_settlement,
            'finish': self.info.offset == len(self.data) - 1
        }

        return observation, reward, terminated, truncated, info


if __name__ == '__main__':
    pass
