import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable, Any


class Env(gym.Env):
    def __init__(self, train: bool = True,
                 observation_space: Any = None, action_space: Any = None,
                 data_getter: Callable = None, data_preprocess: Callable = None, action_decoder: Callable = None):
        super(Env, self).__init__()
        #  action: str = 'discrete',
        self.train = train  # 訓練模式

        if data_getter is None:
            raise ValueError('data_process must be a function')

        self.data_getter = data_getter  # 用以獲取資料
        self.data_preprocess = data_preprocess  # 用以資料前處理 (將返回處理後資料及正規化器)
        self.action_decoder = action_decoder  # 用以解碼動作

        if observation_space is None:
            raise ValueError('observation_space must not be None')

        if action_space is None:
            raise ValueError('action_spaces must not be None')

        self.observation_space = observation_space  # 從外部讀取觀察值空間設定
        self.action_space = action_space  # 從外部讀取動作值空間設定

        ind = pd.read_csv('data/ind.csv')
        self.ind = ind['代號'].to_list()
        self.ind = shuffle(self.ind, random_state=25)

        '''
        資訊類 Container
        '''
        self.now = None
        self.last_close = None
        self.hold = 0  # 持有量
        # self.balance = 1_000_000  # 餘額 (初始為100萬，以每股1000元計算，可以購買1000股，也就是1張)
        self.trade = 1000  # 每次交易量(單位: 股)
        self.holding_count = 0  # 不動作次數
        # self.last_balance = 100_000 #  上一次餘額
        # self.cost = [] #  Deprecated!, 改用平均成本
        self.cost = 0  # 平均成本 (平均成本法)
        self.net = 0
        self.last_net = 0
        self.net_exclude_settlement = 0

        '''
        正規化器
        '''
        self.scaler = None  # set in reset()

    def reset(self, seed=None, *args, **kwargs) -> tuple[np.ndarray, dict]:
        if len(self.ind) == 0:
            ind = pd.read_csv('data/ind.csv')
            self.ind = ind['代號'].to_list()
            self.ind = shuffle(self.ind, random_state=None)
            # return None
        
        stock_num = self.ind.pop(0)

        print(f'reset: {stock_num}')

        '''
        載入個股資料
        '''
        data = self.data_getter(stock_num, is_train=self.train)
        if self.data_preprocess is not None:
            data, self.scaler = self.data_preprocess(data)

        self.now = data

        # 個股改由外部載入

        obs = (self.now[self.now.index == self.now.index[0]]
               .apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y)))
        self.now = self.now.drop(self.now.index[0])
        # self.last_close = obs['Close'].values[0]

        # self.balance = 1_000_000 # 重置餘額
        # self.cost = [] # 重置成本
        
        obs['hold'] = self.hold
        # obs['balance'] = self.balance
        # obs = np.concatenate([obs.to_numpy().astype(np.float32).reshape(-1), np.array([self.hold, self.balance])])
        
        return obs.to_numpy().astype(np.float32), {'stock_id': stock_num}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        buy_or_sell = self.action_decoder(action)

        # 昨天的股價+昨天的買賣超資訊決定今天交易的方向
        obs = (self.now[self.now.index == self.now.index[0]]
               .apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y)))
        self.now = self.now.drop(self.now.index[0])
        
        reward = 0

        last_hold = self.hold
        # last_balance = self.balance

        real_close = self.scaler.inverse_transform(obs['Close'].values[0].reshape(-1, 1))[0][0] \
            if self.scaler is not None else obs['Close'].values[0]

        done = True if len(self.now) == 0 else False

        if done:
            # 強制結算
            # self.balance += real_close * self.hold
            self.net += (real_close - self.cost) * self.hold  # 結算淨損益
            # self.net_exclude_settlement 此處不計算，為了排除非主動交易所造成的損益
            self.hold = 0  # 重置持有量
            self.holding_count = 0  # 重置持平計數器
            self.cost = 0  # 重置平均成本 (不重置也不會有影響，因為持有量為0，下次計算時本次平均成本的權重也為0)

            # if self.balance > 1_000_000:  # 如果最終餘額大於初始餘額，給予獎勵
            #     reward += 10
            if self.net > self.last_net:  # 如果最終淨損益大於0，給予獎勵
                reward += 10

            # self.last_balance = self.balance
            self.last_net = self.net
        else:

            if buy_or_sell is None:  # 持平，不動作
                self.holding_count += 1

                if self.holding_count == 10:
                    reward -= 3
                    self.holding_count = 0
            else:
                self.holding_count = 0  # 持平計數器歸零

                if (not buy_or_sell) & (last_hold == 0):  # 如果無持有，但賣出，懲罰並不動作
                    reward -= 1
                else:  # 正常買入及賣出

                    # 淨損益計算
                    if buy_or_sell:
                        self.cost = (real_close * self.trade + self.cost * self.hold) / (self.trade + self.hold)
                    else:
                        self.net += (real_close - self.cost) * self.trade
                        self.net_exclude_settlement += (real_close - self.cost) * self.trade

                        # 如果有賺，給予獎勵
                        if real_close - self.cost > 0:
                            reward += 5

                    # 持有量及餘額計算
                    self.hold += self.trade if buy_or_sell else -self.trade
                    # self.balance -= real_close * self.trade if buy_or_sell else -real_close * self.trade

                    # 如果破產，懲罰並結束
                    # if self.balance < 0:
                    #     reward -= 10
                    #     done = True

        # print(f'action: {buy_or_sell}, hold: {self.hold}, balance: {self.balance}, reward: {reward}, done: {done}')

        obs['hold'] = self.hold
        # obs['balance'] = self.balance

        return obs.to_numpy().astype(np.float32), reward, done, False, \
            {
                'hold': self.hold,
                'holding_count': self.holding_count,
                'net': self.net,
                'net_exclude_settlement': self.net_exclude_settlement,
                'finish': len(self.ind) == 0
            }


class TensorboardCallback(BaseCallback):
    """
    Tensorboard 記錄
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('env/hold', self.training_env.get_attr('hold')[0])
        # self.logger.record('env/balance', self.training_env.get_attr('balance')[0])
        self.logger.record('env/holding_count', self.training_env.get_attr('holding_count')[0])
        self.logger.record('env/net', self.training_env.get_attr('net')[0])
        self.logger.record('env/net_exclude_settlement', self.training_env.get_attr('net_exclude_settlement')[0])
        return True


if __name__ == '__main__':
    pass
