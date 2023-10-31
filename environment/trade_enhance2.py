from .trade_enhance import InfoContainer, Env as Env_enhance, TensorboardCallback


class Env(Env_enhance):
    def __init__(self, *args, **kwargs):
        super(Env, self).__init__(*args, **kwargs)

    def _calculate_reward(self, return_by_trade: int, terminated: bool = False) -> float:
        if terminated:
            roi = (self.info.balance - self.info.default_balance) / self.info.default_balance  # 已實現報酬率
        else:
            holding_value = self.info.hold * self._locate_data(self.info.offset)['Close']  # unrealized gain/loss
            roi = (self.info.balance + holding_value - self.info.default_balance) / self.info.default_balance  # roi

        reward = roi - self.info.last_roi
        self.info.last_roi = roi

        if terminated:
            pass
        else:
            if return_by_trade == -1:  # 餘額不足導致交易失敗
                if reward >= 0:
                    reward = -5
                else:
                    reward = reward - 5
            elif return_by_trade == 0:  # 買入或賣出時沒賺錢 (獎勵有動作)
                reward = 1
            else:
                reward = reward * 10

        return reward


DESCRIPT = "New action w/ reward by unrealized roi (interact w/ return_by_trade)"


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable!')