from api.works import set_evaluation
from typing import Callable, Optional
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.base_class import BaseAlgorithm
import statistics

from util.get_environment import get_environment


def test(uuid: str,
         model: BaseAlgorithm,
         data_locator: Callable,
         random_state: Optional[int] = None):
    env, callback = get_environment('trade_enhance')
    env = env(data_locator=data_locator, data_root='data/test', random_state=random_state)
    writer = SummaryWriter(f'tasks/works/{uuid}/{uuid}_test')

    obs, info = env.reset()

    rois = []

    step_count = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)

        writer.add_scalar('env/balance', env.info.balance, step_count)
        writer.add_scalar('env/hold', env.info.hold, step_count)
        writer.add_scalar('env/holding_count', env.info.holding_count, step_count)
        writer.add_scalar('env/net', env.info.net, step_count)
        writer.add_scalar('env/reward', rewards, step_count)

        step_count += 1

        if terminated:
            roi = (env.info.balance - env.info.default_balance) / env.info.default_balance
            writer.add_scalar('env/roi', roi, step_count)
            set_evaluation(uuid, {
                'name': env.data_locator.get_index(),
                'value': roi
            })
            rois.append(roi)
            obs, info = env.reset()

            if env.data_locator.offset == len(env.data_locator.index) - 1:  # means done all stocks
                set_evaluation(uuid, {
                    'name': 'roi_mean',
                    'value': statistics.mean(rois)
                })
                set_evaluation(uuid, {
                    'name': 'roi_median',
                    'value': statistics.median(rois)
                })
                set_evaluation(uuid, {
                    'name': 'roi_max',
                    'value': max(rois)
                })
                set_evaluation(uuid, {
                    'name': 'roi_min',
                    'value': min(rois)
                })
                break
