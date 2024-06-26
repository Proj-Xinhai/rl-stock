import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from math import floor
from typing import Optional, Tuple, Callable


class InfoContainer(object):
    def __init__(self, default_balance: int = 1_000_000):
        super(InfoContainer, self).__init__()
        self.default_balance = default_balance
        self.offset = 0  # 資料定位
        self.balance = self.default_balance  # 現金餘額
        self.net = 0  # 淨損益
        self.cost = 0  # 平均成本
        self.cost_total = 0  # 總成本
        self.hold = 0  # 持有量
        self.holding_count = 0  # 不動作次數
        self.last_roi = 0  # 上一次售出的報酬率

    def reset(self):
        self.balance = self.default_balance
        self.cost = 0
        self.hold = 0
        self.holding_count = 0
        self.last_roi = 0


class Env(gym.Env):
    def __init__(self,
                 data_locator: Callable,
                 index_path: str = 'data/ind.csv',
                 data_root: str = 'data',
                 start: str = '2018-02-21',  # train
                 end: str = '2023-01-17',  # train
                 online: bool = False,
                 stock_id: Optional[list] = None,
                 default_balance: int = 1_000_000,
                 random_state: Optional[int] = None):
        """
        """
        super(Env, self).__init__()

        """
        資訊類 Container:
        data_getter: 資料取樣器
        data: 當前資料 (當下 episode 的資料)
        info: 資訊容器
        """
        self.index_path = index_path
        self.data_root = data_root
        self.random_state = random_state
        self.data_locator = data_locator(index_path=self.index_path,
                                         data_root=self.data_root,
                                         start=start,
                                         end=end,
                                         online=online,
                                         stock_id=stock_id,
                                         random_state=self.random_state)
        self.data = self.data_locator.next()
        self.info = InfoContainer(default_balance=default_balance)

        """
        """
        # 直接用傳入的資料取樣
        sample = self._locate_data(0).to_numpy().astype(np.float32).shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)

    def _decode_action(self, action: np.float32) -> Tuple[Optional[bool], int]:
        # action 是比例
        if action > 0:  # Buy
            project = self.info.balance * action  # 愈買入金額
            trade = project / self.data.iloc[self.info.offset]['close']  # 實際買入股數
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
        data = self.data.iloc[offset].copy()
        data['balance'] = self.info.balance  # 把餘額加入
        data['hold'] = self.info.hold  # 把持有量加入
        return data

    def reset(self, seed=None, *args, **kwargs) -> tuple[np.ndarray, dict]:
        # 理論上，這裡的 observation 會與重置前的 observation 相同 (offset 相同)
        print(f'reset: {self.data_locator.get_index()}')

        self.info.reset()  # 重置部分資訊

        observation = self._locate_data(self.info.offset)
        info = {
            'stock_num': self.data_locator.get_index(),
        }

        return observation.to_numpy().astype(np.float32), info

    def _calculate_trade(self, buy_or_sell: bool, trade: int) -> int:
        if buy_or_sell is None:  # Hold
            self.info.holding_count += 1

            if self.info.holding_count >= 10:  # 連續第 10 天沒動作
                pass  # 有可能影響訓練，先不懲罰做觀察

            return 0

        self.info.holding_count = 0  # 重置不動作次數

        if trade == 0:  # 因餘額不足 (或剩餘持有量不足) 導致的無法交易
            return -1  # 懲罰

        # Buy or Sell
        if buy_or_sell:  # Buy
            price = self._locate_data(self.info.offset)['close']  # 買入單價
            # 小數點以下應捨去
            # 計算加權平均成本
            self.info.cost = (floor(self.info.cost * self.info.hold) + floor(price * trade)) / (self.info.hold + trade)
            self.info.balance -= floor(price * trade)  # 扣除買入金額
            self.info.cost_total += floor(price * trade)  # 累計成本
            self.info.hold += trade  # 持有量增加
            return 0  # TODO: 嘗試有動作給予獎勵
        else:  # Sell
            price = self._locate_data(self.info.offset)['close']  # 賣出單價
            # 加權平均成本 (self.info.cost) 不會改變
            # 小數點以下應捨去
            self.info.balance += floor(price * trade)  # 加回賣出金額
            self.info.hold -= trade  # 持有量減少
            self.info.net = (self.info.balance + floor(self.info.cost * self.info.hold)
                             - self.info.default_balance)  # 損益為餘額減去初始餘額 (當次episode累計損益)

            roi = (self.info.balance - self.info.default_balance) / self.info.default_balance

            if roi > self.info.last_roi:
                self.info.last_roi = roi
                return 5  # 有賺錢所以獎勵
            else:
                self.info.last_roi = roi
                return 0  # 沒賺錢，不給獎勵  # TODO: 嘗試有動作給予獎勵

    def _calculate_terminated(self, return_by_trade: int) -> int:
        if self.info.hold > 0:
            price = self._locate_data(self.info.offset)['close']  # 賣出單價
            # 小數點以下應捨去
            self.info.balance += floor(price * self.info.hold)  # 加回餘額
            self.info.net = self.info.balance - self.info.default_balance  # 損益為餘額減去初始餘額 (當次episode累計損益)

            roi = (self.info.balance - self.info.default_balance) / self.info.default_balance

            if roi > 0:
                if return_by_trade == 0:
                    reward = 3  # 如果最終報酬率為正，且最後一次交易為持平，給予較少獎勵
                else:
                    reward = 8  # 如果最終報酬率為正，且最後一次交易非持平，給予較高獎勵
            else:
                reward = -1  # 如果最終報酬率為負，給予懲罰

        else:
            roi = (self.info.balance - self.info.default_balance) / self.info.default_balance
            if roi > 0:
                reward = 10  # 如果最終報酬率為正，給予較高獎勵
            else:
                reward = -5  # 如果最終報酬率為負，給予懲罰

        return reward

    def _calculate_reward(self, return_by_trade: int, terminated: bool = False) -> float:
        if terminated:
            roi = (self.info.balance - self.info.default_balance) / self.info.default_balance  # 已實現報酬率
        else:
            holding_value = self.info.hold * self._locate_data(self.info.offset)['close']  # unrealized gain/loss
            roi = (self.info.balance + holding_value - self.info.default_balance) / self.info.default_balance  # roi

        reward = roi - self.info.last_roi
        self.info.last_roi = roi

        return reward

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        buy_or_sell, trade = self._decode_action(action)

        reward = self._calculate_trade(buy_or_sell, trade)
        terminated = True if self.info.offset == len(self.data) - 1 else False
        truncated = False  # 這表示因超時導致的終止，我們不會有這種情況

        if terminated:  # 已經結束這檔股票的交易，進行結算
            reward = self._calculate_terminated(reward)  # 結算

            self.data = self.data_locator.next()  # 獲取資料
            self.info.offset = 0  # 資料定位歸零
        else:
            self.info.offset += 1  # 下移資料定位 (換日)

        # TODO: reward 要與 _calculate_trade 的 return 配合
        reward = self._calculate_reward(reward, terminated)

        observation = self._locate_data(self.info.offset).to_numpy().astype(np.float32)
        action_name = 'hold'
        if buy_or_sell:
            action_name = 'buy'
        else:
            action_name = 'sell'

        info = {
            'date': self._locate_data(self.info.offset - 1).name,
            'action': action_name,
            'trade': trade,
        }

        return observation, reward, terminated, truncated, info


class TensorboardCallback(BaseCallback):
    """
    Tensorboard 記錄
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        info = self.training_env.get_attr('info')[0]
        rollback_cost = info.hold * info.cost  # 把持有成本加回去
        unrealized_gain_loss = info.hold * self.training_env.get_attr('data')[0].iloc[info.offset]['close']  # 未實現損益
        # 未實現損益 - 持有成本 = 未實現淨損益 (期中任何時機皆可)
        # 帳面餘額 - 期初餘額 = 已實現淨損益 (期末，此時應不存在持有成本或未實現損益)
        # roi 為 (帳面餘額 + 持有成本 - 期初餘額) / 期初餘額 = 已實現報酬率
        roi = (info.balance + rollback_cost - info.default_balance) / info.default_balance
        # roi_unrealized 為 (帳面餘額 + 未實現損益 - 期初餘額) / 期初餘額 = 已實現報酬率
        roi_unrealized = (info.balance + unrealized_gain_loss - info.default_balance) / info.default_balance
        self.logger.record('env/balance', info.balance)
        self.logger.record('env/hold', info.hold)
        self.logger.record('env/holding_count', info.holding_count)
        self.logger.record('env/net', info.net)
        self.logger.record('env/roi', roi)
        self.logger.record('env/roi_unrealized', roi_unrealized)
        return True


DESCRIPT = "New action w/ reward by unrealized roi"


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable!')
