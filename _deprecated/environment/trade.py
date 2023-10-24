import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from stable_baselines3.common.callbacks import BaseCallback


class Env(gym.Env):
    def __init__(self, train: bool = True, action: str = 'discrete'):
        super(Env, self).__init__()

        self.train = train
        self.action = action

        self.observation_space = spaces.Box(
            low=np.array([np.concatenate([np.zeros(7), np.zeros(4), -np.ones(7), np.array([-np.inf])])]),
            high=np.array([np.concatenate([np.ones(7), np.ones(4), np.ones(7), np.array([np.inf])])]),
            shape=(1, 7 + 4 + 7 + 1),  # 7: 個股資料, 4: SMA, 7: 法人買賣超, 1: 持有量
            dtype=np.float32
        )  # 觀察值範圍處理
        if self.action == 'discrete':
            self.action_space = spaces.Discrete(3)  # buy or sell or hold
        elif self.action == 'box':
            self.action_space = spaces.Box(low=np.float32(-1), high=np.float32(1), shape=(1,), dtype=np.float32)
        ind = pd.read_csv('data/ind.csv')
        self.ind = ind['代號'].to_list()
        self.ind = shuffle(self.ind, random_state=25)

        if self.action == 'box':
            self.last_action = 0

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
        self.scaler = MinMaxScaler()

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
        data_path = 'data/train' if self.train else 'data/test'
        # 個股資料
        s = pd.read_csv(f'{data_path}/個股/{stock_num}.csv').reset_index(drop=True).set_index('Date')
        # 三大法人買賣超
        i = pd.read_csv(f'{data_path}/法人買賣超日報_個股/{stock_num}.csv', index_col=0).reset_index(drop=True).set_index('Date')
        # 平均線
        s['MA5'] = talib.MA(s['Close'], timeperiod=5)
        s['MA10'] = talib.MA(s['Close'], timeperiod=10)
        s['MA20'] = talib.MA(s['Close'], timeperiod=20)
        s['MA60'] = talib.MA(s['Close'], timeperiod=60)

        '''
        正規化法人買賣超，且只保留買賣超部分
        正規化方式: 1: 買超, -1: 賣超, 0: 無
        '''
        i = i.apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        i = i[['外陸資買賣超股數(不含外資自營商)', '外資自營商買賣超股數', '投信買賣超股數', '自營商買賣超股數', '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)', '三大法人買賣超股數']]
        i = i.apply(lambda x: x.apply(lambda y: 1 if float(y) > 0 else -1 if float(y) < 0 else 0))
        # i = i.shift(1) # 往下偏移一天 (因為當天結束才會統計買賣超資訊，實務上交易日當天是不知道當天買賣超資訊的)
        # i = i.fillna(0)

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        self.now = pd.concat([s, i], axis=1, sort=True).dropna()

        # 正規化個股資料 (每股皆進行一次)
        self.scaler.fit(self.now['Close'].values.reshape(-1, 1))
        self.now['Open'] = self.scaler.transform(self.now['Open'].values.reshape(-1, 1)).reshape(-1)
        self.now['High'] = self.scaler.transform(self.now['High'].values.reshape(-1, 1)).reshape(-1)
        self.now['Low'] = self.scaler.transform(self.now['Low'].values.reshape(-1, 1)).reshape(-1)
        self.now['Close'] = self.scaler.transform(self.now['Close'].values.reshape(-1, 1)).reshape(-1)
        self.now['MA5'] = self.scaler.transform(self.now['MA5'].values.reshape(-1, 1)).reshape(-1)
        self.now['MA10'] = self.scaler.transform(self.now['MA10'].values.reshape(-1, 1)).reshape(-1)
        self.now['MA20'] = self.scaler.transform(self.now['MA20'].values.reshape(-1, 1)).reshape(-1)
        self.now['MA60'] = self.scaler.transform(self.now['MA60'].values.reshape(-1, 1)).reshape(-1)

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
        buy_or_sell = None
        if self.action == 'discrete':
            buy_or_sell = bool(int(action)) if action != 2 else None
        elif self.action == 'box':
            buy_or_sell = True if action > self.last_action else False if action < self.last_action else None
            self.last_action = action

        # 昨天的股價+昨天的買賣超資訊決定今天交易的方向
        obs = (self.now[self.now.index == self.now.index[0]]
               .apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y)))
        self.now = self.now.drop(self.now.index[0])
        
        reward = 0

        last_hold = self.hold
        # last_balance = self.balance

        real_close = self.scaler.inverse_transform(obs['Close'].values[0].reshape(-1, 1))[0][0]

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
