# from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
import datetime

from deprecated.environment.trade import Env, TensorboardCallback

if __name__ == '__main__':
    env = Env()

    start = datetime.datetime.now()
    print(f'start: {start}')

    model = RecurrentPPO('MlpLstmPolicy', env, learning_rate=3e-4, verbose=1,
                         tensorboard_log='./tensorboard/ppo_trade')  # , n_steps=1205
    model.learn(total_timesteps=1_000_000, progress_bar=True, callback=TensorboardCallback())

    model.save('model/trade_recurrent_ppo_3')

    end = datetime.datetime.now()

    print(f'time: {end - start}')
    print(f'start: {start}')
    print(f'end: {end}')
