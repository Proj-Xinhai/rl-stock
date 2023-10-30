import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from math import floor
from typing import Optional, Tuple, Callable
from environment.trade_enhance import InfoContainer, TensorboardCallback, Env as Env_enhance


class Env(Env_enhance):
    def __init__(self, *args, **kwargs):
        """
        """
        super(Env, self).__init__(*args, **kwargs)

    def _calculate_reward(self, return_by_trade: int, terminated: bool = False) -> float:
        if terminated:
            reward = (self.info.balance - self.info.default_balance) / self.info.default_balance  # 已實現報酬率
        else:
            holding_cost = self.info.hold * self.info.cost  # holding cost  # realized gain/loss only
            reward = (self.info.balance + holding_cost - self.info.default_balance) / self.info.default_balance  # roi

        return reward


DESCRIPT = "New action w/ reward by realized roi"


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable!')
