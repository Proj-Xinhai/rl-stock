import importlib
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Optional


def get_algorithm(algorithm: str) -> Optional[BaseAlgorithm]:
    if algorithm in importlib.import_module('stable_baselines3').__dict__:
        return importlib.import_module('stable_baselines3').__dict__[algorithm]
    elif algorithm in importlib.import_module('sb3_contrib').__dict__:
        return importlib.import_module('sb3_contrib').__dict__[algorithm]
    else:
        return None


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
