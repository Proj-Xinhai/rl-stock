from stable_baselines3 import DQN
import datetime

from environment.trade import Env, TensorboardCallback

MODE = 0  # 0: train, 1: test

if __name__ == '__main__':

    if MODE == 0:
        env = Env()

        start = datetime.datetime.now()
        print(f'start: {start}')

        model = DQN('MlpPolicy', env, learning_rate=3e-4, verbose=1,
                             tensorboard_log='./tensorboard/dqn_trade')  # , n_steps=1205
        model.learn(total_timesteps=1_000_000, progress_bar=True, callback=TensorboardCallback())

        model.save('model/trade_dqn_1')

        end = datetime.datetime.now()

        print(f'time: {end - start}')
        print(f'start: {start}')
        print(f'end: {end}')

    elif MODE == 1:
        from torch.utils.tensorboard import SummaryWriter
        import os

        env = Env(train=False)

        model = DQN.load('model/trade_dqn_1')

        tb_dir = 'tensorboard/dqn_trade1_evaluate'
        # if not os.path.isdir(tb_dir):
        #     os.mkdir(tb_dir)

        obs, info = env.reset()

        writer = SummaryWriter(os.path.join(tb_dir, info['stock_id']))

        step_count = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)

            writer.add_scalar('env/hold', info['hold'], step_count)
            # writer.add_scalar('env/balance', info['balance'], step_count)
            writer.add_scalar('env/holding_count', info['holding_count'], step_count)
            writer.add_scalar('env/net', info['net'], step_count)
            writer.add_scalar('env/net_exclude_settlement', info['net_exclude_settlement'], step_count)

            step_count += 1

            if terminated:
                writer.close()
                if info['finish']:
                    break
                obs, info = env.reset()
                writer = SummaryWriter(os.path.join(tb_dir, info['stock_id']))
                step_count = 0
                # env.balance = 1_000_000 # 重置餘額
                env.net = 0  # 重置淨損益
                env.net_not_include_settlement = 0  # 重置淨損益(不含結算)

    else:
        raise ValueError('MODE must be 0 or 1')
