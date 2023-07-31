from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
import datetime

from environment.trade import Env, TensorboardCallback

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    import os
    env = Env(train=False)
    
    tb_dir = 'tensorboard/random_trade'

    obs, info = env.reset()
    
    writer = SummaryWriter(os.path.join(tb_dir, info['stock_id']))
    
    step_count = 0
    while True:
        action = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(action)

        writer.add_scalar('env/hold', info['hold'], step_count)
        # writer.add_scalar('env/balance', info['balance'], step_count)
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
            # env.balance = 1_000_000 # 重置餘額
            env.net = 0 # 重置淨損益
            env.net_not_include_settlement = 0 # 重置淨損益(不含結算)
