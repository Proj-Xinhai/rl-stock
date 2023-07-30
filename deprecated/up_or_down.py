'''
This file is a copy of deprecated file /environment/up_or_down.py
'''
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

class Env(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=np.array([np.concatenate([np.zeros(7), -np.ones(7)])]), high=np.array([np.concatenate([np.ones(7), np.ones(7)])]), shape=(1, 7 + 7), dtype=np.float32) # 觀察值範圍處理
        self.action_space = spaces.Discrete(9) # up or down
        ind = pd.read_csv('ind.csv')
        self.ind = ind['代號'].to_list()
        self.ind = shuffle(self.ind, random_state=25)

        self.now = None
        self.last_close = None

        self.scaler = MinMaxScaler()

    def reset(self, seed=None):
        if len(self.ind) == 0:
            return None
        else:
            stock_num = self.ind.pop(0)

        print(f'reset: {stock_num}')

        '''
        載入個股資料
        '''
        s = pd.read_csv(f'data/train/個股/{stock_num}.csv').reset_index(drop=True).set_index('Date')
        i = pd.read_csv(f'data/train/法人買賣超日報_個股/{stock_num}.csv', index_col=0).reset_index(drop=True).set_index('Date')

        '''
        正規化法人買賣超，且只保留買賣超部分
        正規化方式: 1: 買超, -1: 賣超, 0: 無
        '''
        i = i.apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        i = i[['外陸資買賣超股數(不含外資自營商)', '外資自營商買賣超股數', '投信買賣超股數', '自營商買賣超股數', '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)', '三大法人買賣超股數']]
        i = i.apply(lambda x: x.apply(lambda y: 1 if float(y) > 0 else -1 if float(y) < 0 else 0))
        # i = i.shift(1) # 往下偏移一天
        # i = i.fillna(0)

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        self.now = pd.concat([s, i], axis=1, sort=True).dropna()

        # 正規化個股資料 (每股皆進行一次)
        self.scaler.fit(self.now['Close'].values.reshape(-1, 1))
        self.now['Close'] = self.scaler.transform(self.now['Close'].values.reshape(-1, 1)).reshape(-1)

        obs = self.now[self.now.index == self.now.index[0]].apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        self.now = self.now.drop(self.now.index[0])
        self.last_close = obs['Close'].values[0]
        
        return obs.to_numpy().astype(np.float32), {}

    def step(self, action):
        pred_up_or_down = bool(action > 4) # 小於4為下跌，大於4為上漲: 40%機率為上漲，60%機率為下跌 (不動去哪了????)

        obs = self.now[self.now.index == self.now.index[0]].apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        self.now = self.now.drop(self.now.index[0])
        
        reward = 0
        real_up_or_down = True if obs['Close'].values[0] > self.last_close else False

        if pred_up_or_down == real_up_or_down:
            reward = 2
        elif (not pred_up_or_down) & real_up_or_down:
            reward = -2
        elif pred_up_or_down ^ real_up_or_down:
            reward = -1

        
        self.last_close = obs['Close'].values[0]


        done = True if len(self.now) == 0 else False

        print(f'pred: {pred_up_or_down}, real: {real_up_or_down}, reward: {reward}, done: {done}')

        return obs.to_numpy().astype(np.float32), reward, done, False, {}

if __name__ == '__main__':
    pass