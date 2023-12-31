'''
This file is a copy of deprecated version of /sb_recurrent_ppo_trade.py
This version contains balance which is not used in new version
'''
from sb3_contrib import RecurrentPPO
import datetime

from deprecated.environment.trade import Env, TensorboardCallback

MODE = 1 # 0: train, 1: test

if __name__ == '__main__':

    if MODE == 0:
        env = Env()

        start = datetime.datetime.now()
        print(f'start: {start}')


        model = RecurrentPPO('MlpLstmPolicy', env, learning_rate=3e-4, verbose=1, tensorboard_log='./ppo_trade') # , n_steps=1205
        model.learn(total_timesteps=1_000_000, progress_bar=True, callback=TensorboardCallback())

        model.save('trade_recurrent_ppo_2')

        end = datetime.datetime.now()

        print(f'time: {end - start}')
        print(f'start: {start}')
        print(f'end: {end}')

    elif MODE == 1:
        from torch.utils.tensorboard import SummaryWriter
        import os
        env = Env(train=False)

        model = RecurrentPPO.load('trade_recurrent_ppo_2')
        
        tb_dir = 'ppo_trade2_evaluate'
        # if not os.path.isdir(tb_dir):
        #     os.mkdir(tb_dir)

        obs, info = env.reset()
        
        writer = SummaryWriter(os.path.join(tb_dir, info['stock_id']))
        
        step_count = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)

            writer.add_scalar('env/hold', info['hold'], step_count)
            writer.add_scalar('env/balance', info['balance'], step_count)
            writer.add_scalar('env/holding_count', info['holding_count'], step_count)
            writer.add_scalar('env/net', info['net'], step_count)
            writer.add_scalar('env/net_not_include_settlement', info['net_not_include_settlement'], step_count)
            
            step_count += 1
            
            if terminated:
                writer.close()
                if info['finish']:
                    break
                obs, info = env.reset()
                writer = SummaryWriter(os.path.join(tb_dir, info['stock_id']))
                step_count = 0
                env.balance = 1_000_000 # 重置餘額
                env.net = 0 # 重置淨損益
                env.net_not_include_settlement = 0 # 重置淨損益(不含結算)

    else:
        raise ValueError('MODE must be 0 or 1')
