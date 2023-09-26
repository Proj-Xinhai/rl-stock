from stable_baselines3 import DQN
from util.pipeline import Pipeline

from NormalAndSMAHelper import NormalHelper, SMAHelper


if __name__ == '__main__':
    # tasks to run
    tasks = [
        {
            'name': 'trade_recurrent_ppo',
            # algorithm
            'algorithm': DQN,
            'algorithm_args': {
                'policy': 'MlpLstmPolicy',
                'learning_rate': 3e-4
            },
            'learn_args': {
                'total_timesteps': 1_000_000
            },
            # environment
            'helper': NormalHelper,
        },
        {
            'name': 'trade_recurrent_ppo_sma',
            # algorithm
            'algorithm': DQN,
            'algorithm_args': {
                'policy': 'MlpLstmPolicy',
                'learning_rate': 3e-4
            },
            'learn_args': {
                'total_timesteps': 1_000_000
            },
            # environment
            'helper': SMAHelper,
        }
    ]

    p = Pipeline(tasks)  # init pipeline

    train = p.train()  # get train generator
    test = p.test()  # get test generator

    next(train)  # init train
    next(test)  # init test

    while (t := next(train)) is not None:  # do train
        next(test)  # do test
        print(f'task {t} finished')
