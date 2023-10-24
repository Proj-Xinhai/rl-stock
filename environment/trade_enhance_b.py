from typing import Optional, Callable

from environment.trade_enhance import (InfoContainer as foreign_InfoContainer,
                                       Env as foreign_Env,
                                       TensorboardCallback as foreign_TensorboardCallback)


class InfoContainer(foreign_InfoContainer):
    def __init__(self, default_balance: int = 1_000_000):
        super(InfoContainer, self).__init__(default_balance=default_balance)


class Env(foreign_Env):
    def __init__(self,
                 data_locator: Callable,
                 index_path: str = 'data/ind.csv',
                 data_root: str = 'data/test',
                 random_state: Optional[int] = None):
        """
        """
        super(Env, self).__init__(data_locator=data_locator,
                                  index_path=index_path,
                                  data_root=data_root,
                                  random_state=random_state)

    def _calculate_reward(self, return_by_trade: int, terminated: bool = False) -> float:
        if terminated:
            reward = (self.info.balance - self.info.default_balance) / self.info.default_balance  # 已實現報酬率
            if return_by_trade == 0:
                reward = 0
            elif return_by_trade == 3:
                reward = reward * 0.5
            elif return_by_trade == 8:
                reward = reward  # 這裡的 reward 應該是正的
            elif return_by_trade == 10:
                reward = reward * 2
            elif reward == -5:
                reward = reward  # 這裡的 reward 應該是負的
            else:
                raise ValueError(f'Unknown return_by_trade: {return_by_trade}! This should not happen!')
        else:
            holding_value = self.info.hold * self._locate_data(self.info.offset)['Close']  # unrealized gain/loss
            reward = (self.info.balance + holding_value - self.info.default_balance) / self.info.default_balance  # roi

            if return_by_trade == 0:
                reward = 0
            elif return_by_trade == 5:
                reward = reward
            elif return_by_trade == -1:
                if reward > 0:
                    reward = reward * 0.5
                else:
                    reward = reward * 2
            else:
                raise ValueError(f'Unknown return_by_trade: {return_by_trade}! This should not happen!')

        return reward


class TensorboardCallback(foreign_TensorboardCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)


DESCRIPT = "New action w/ reward by unrealized roi (interact w/ return by trade)"


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable!')
