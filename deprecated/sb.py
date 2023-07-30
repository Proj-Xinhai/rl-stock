'''
This file is a copy of deprecated file /sb.py
'''
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd

class ENV(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32) # 觀察值範圍處理 + 17
        self.action_space = spaces.Discrete(20) # +-10%

        ind = pd.read_csv('ind.csv')
        self.ind = ind['代號'].to_list()

        self.now = None
        self.last_close = None

    def reset(self, seed=None):
        if len(self.ind) == 0:
            return None
        else:
            stock_num = self.ind.pop(0)

        print(f'reset: {stock_num}')

        s = pd.read_csv(f'data/train/個股/{stock_num}.csv').reset_index(drop=True).set_index('Date')
        # i = pd.read_csv(f'data/train/法人買賣超日報_個股/{stock_num}.csv', index_col=0).reset_index(drop=True).set_index('Date')
        # self.now = pd.concat([s, i], axis=1, sort=True).dropna()
        self.now = s.dropna()

        obs = self.now[self.now.index == self.now.index[0]].apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        self.now = self.now.drop(self.now.index[0])
        self.last_close = obs['Close'].values[0]
        
        return obs.to_numpy().astype(np.float32), {}

    def step(self, action):
        action = action - 10
        pred_close = self.last_close * (1 + action / 100)

        obs = self.now[self.now.index == self.now.index[0]].apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        self.now = self.now.drop(self.now.index[0])
        self.last_close = obs['Close'].values[0]
        
        reward = 0
        deviation = abs(pred_close - self.last_close) / self.last_close
        if deviation == 0:
            reward = 10
        elif deviation <= 0.01:
            reward = -1
        elif deviation <= 0.03:
            reward = -5
        elif deviation <= 0.05:
            reward = -10
        elif deviation <= 0.07:
            reward = -20
        elif deviation <= 0.1:
            reward = -50
        else:
            reward = -100


        done = True if len(self.now) == 0 else False

        print(f'pred_close: {pred_close}({action}%), true_close: {self.last_close}, reward: {reward}, error: {deviation}, done: {done}')

        return obs.to_numpy().astype(np.float32), reward, done, False, {}

if __name__ == '__main__':
    from stable_baselines3 import PPO, A2C
    import datetime
    env = ENV()

    start = datetime.datetime.now()
    print(f'start: {start}')

    '''
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    model.save('ppo')

    end = datetime.datetime.now()

    print(f'time: {end - start}')
    print(f'start: {start}')
    print(f'end: {end}')
    '''

    '''
    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    model.save('a2c')

    end = datetime.datetime.now()

    print(f'time: {end - start}')
    print(f'start: {start}')
    print(f'end: {end}')
    '''

    '''
    model = PPO.load('ppo')

    for i in range(506):
        obs, _ = env.reset()
    for i in range(1500):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, truncated, info = env.step(action)
        if dones:
            obs, _ = env.reset()
        # env.render()
    '''

    
    model = A2C.load('a2c')

    for i in range(789):
        obs, _ = env.reset()
    for i in range(1500):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = env.step(action)
        if dones:
            obs, _ = env.reset()
        # env.render()
    

    # obs = env.reset()
    # for i in range(12000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = env.step(action)
    #     if dones:
    #         obs = env.reset()
    #     # env.render()