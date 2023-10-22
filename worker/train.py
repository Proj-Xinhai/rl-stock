from environment.trade_enhance import Env, TensorboardCallback
from typing import Callable, Optional
from stable_baselines3.common.base_class import BaseAlgorithm


def train(uuid: str,
          data_locator: Callable,
          algorithm: Callable,
          algorithm_args: dict,
          learn_args: dict,
          random_state: Optional[int] = None) -> BaseAlgorithm:
    env = Env(data_locator=data_locator, data_root='data/train', random_state=random_state)

    model = algorithm(**algorithm_args, env=env, verbose=1, tensorboard_log=f'tasks/works/{uuid}')
    model.learn(**learn_args, progress_bar=True, callback=TensorboardCallback(), tb_log_name=f'{uuid}')
    model.save(f'tasks/works/{uuid}/{uuid}')

    return model
