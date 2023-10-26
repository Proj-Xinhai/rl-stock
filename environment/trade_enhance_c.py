from typing import Optional, Callable

from environment.trade_enhance import (InfoContainer as foreign_InfoContainer,
                                       Env as foreign_Env,
                                       TensorboardCallback as foreign_TensorboardCallback)


class InfoContainer(foreign_InfoContainer):
    def __init__(self, default_balance: int = 1_000_000):
        super(InfoContainer, self).__init__(default_balance=default_balance)

    def reset(self):
        self.default_balance = self.balance  # 將預設餘額設為當前餘額
        super(InfoContainer, self).reset()


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

        self.info = InfoContainer()

    def _calculate_reward(self, return_by_trade: int, terminated: bool = False) -> float:
        if terminated:
            reward = (self.info.balance - self.info.default_balance) / self.info.default_balance  # 已實現報酬率
        else:
            holding_value = self.info.hold * self._locate_data(self.info.offset)['Close']  # unrealized gain/loss
            roi = (self.info.balance + holding_value - self.info.default_balance) / self.info.default_balance  # roi
            self.info.last_roi = roi
            reward = roi - self.info.last_roi

        return reward


class TensorboardCallback(foreign_TensorboardCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)


DESCRIPT = "New action w/ reward by difference of unrealized roi"


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable!')
