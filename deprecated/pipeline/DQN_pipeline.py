from stable_baselines3 import DQN
from deprecated.pipeline import Pipeline

from NormalAndSMAHelper import SMAHelper


if __name__ == '__main__':
    # tasks to run
    tasks = [
        # {
        #     'name': 'trade_dqn',
        #     # algorithm
        #     'algorithm': DQN,
        #     'algorithm_args': {
        #         'policy': 'MlpPolicy',
        #         'learning_rate': 1e-4
        #     },
        #     'learn_args': {
        #         'total_timesteps': 1_000_000
        #     },
        #     # environment
        #     'helper': NormalHelper,
        # },
        {
            'name': 'trade_dqn_sma',
            # algorithm
            'algorithm': DQN,
            'algorithm_args': {
                'policy': 'MlpPolicy',
                'learning_rate': 1e-4
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
