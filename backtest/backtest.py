from typing import Tuple, Optional
from api.works import load_work
from api.tasks import load_task
from api.environments import get_environment
import numpy as np
import pandas as pd

def backtest(stock: str, model: str, default_balance: int, start: str, end: str):
    try:
        work = load_work(model)
        task = load_task(work['task_name'])
        algorithm = task['algorithm']
        data_locator = task['data_locator']
        environment = task['environment']

        env, _ = get_environment(environment)
        env = env(data_locator=data_locator,
                  data_root='data',
                  start=start,
                  end=end,
                  online=True,
                  stock_id=[stock],
                  default_balance=default_balance,
                  random_state=None)

        model = algorithm.load(f'tasks/works/{model}/{model}')

        obs, info = env.reset()

        actions = pd.DataFrame(columns=['date', 'action', 'balance', 'hold', 'net', 'reward'])

        # for recurrent model
        state = None
        episode_start = np.ones(1, dtype=bool)

        step_count = 0
        while True:
            if algorithm.__name__ == 'RecurrentPPO':
                action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, rewards, terminated, truncated, info = env.step(action)

            actions = pd.concat([actions, pd.DataFrame({
                'date': [str(info['date'])],
                'original_action': [action.item(0)],
                'action': [info['action']],
                'trade': [info['trade']],
                'balance': [env.info.balance],
                'hold': [env.info.hold],
                'net': [env.info.net],
                'reward': [rewards],
            })])

            # for recurrent model
            episode_start = terminated

            step_count += 1

            if terminated:
                roi = (env.info.balance - env.info.default_balance) / env.info.default_balance
                return True, 'success', {
                    'name': env.data_locator.get_index(),
                    'value': roi,
                    'actions': actions.to_dict(orient='records')
                }
    except Exception as e:
        return False, str(e), None



if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
