import os
import importlib
from typing import Optional, Tuple, Callable


def get_environment(environment: str) -> Tuple[Optional[Callable], Optional[Callable]]:
    if not os.path.exists(f'environment/{environment}.py'):
        return None, None

    env = importlib.import_module('environment.' + environment).__dict__['Env']
    callback = importlib.import_module('environment.' + environment).__dict__['TensorboardCallback']
    return env, callback


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
