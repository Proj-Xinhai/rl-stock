import os
import re
import json
from datetime import datetime
from typing import Tuple, Optional
import importlib
from pathvalidate import sanitize_filename

from api.util.load_task import load_task
from api.list_data_locator import list_data_locator
from api.util.eval_args import eval_args
from api.util.get_algorithm import get_algorithm
from environment.trade_enhance import Env  # TODO: maybe can choose env by setting


def create_task(name: str,
                algorithm: str,
                algorithm_args: dict,
                learn_args: dict,
                data_locator: str,
                random_state: Optional[int]) -> Tuple[bool, str, str]:
    # check if name is valid
    if name == '':
        return False, 'name', f'cannot be empty'

    _, _, error, detail = eval_args(algorithm, algorithm_args, learn_args)

    if error is not None:
        return False, error, detail

    # check if data locator is valid (in general, this will not be invalid)
    if next((loc for loc in list_data_locator() if loc['name'] == data_locator), None) is None:
        return False, 'data_locator', f'`{data_locator}` not found'

    args = {
        'name': name,  # str
        'algorithm': algorithm,  # class name like A2C
        'algorithm_args': algorithm_args,  # dict
        'learn_args': learn_args,  # dict
        'data_locator': data_locator,  # .py file
        'random_state': None if random_state == '' else random_state  # int or null
    }

    # check if name have invalid characters
    name = sanitize_filename(name)
    # first check if task already exists, if so, add number
    if os.path.exists(f'tasks/{name}.json'):
        i = 1
        while os.path.exists(f'tasks/{name}_{i}.json'):
            i += 1
        name = f'{name}_{i}'
        args['name'] = name

    # try to inistialize task
    try:
        sb3 = get_algorithm(algorithm)
        if sb3 is None:
            return False, 'algorithm', 'algorithm not found'
        else:
            loc = importlib.import_module('data_locator.' + data_locator).__dict__['EXPORT']
            env = Env(data_locator=loc)
            sb3(**algorithm_args, env=env)
    except Exception as e:
        return False, 'algorithm_args', str(e)

    # write to file
    with open(f'tasks/{name}.json', 'w') as f:
        json.dump(args, f)

    return True, 'success', name


def remove_task(name: str) -> Tuple[bool, str, str]:
    if os.path.exists(f'tasks/{name}.json'):
        os.remove(f'tasks/{name}.json')
        return True, 'success', name
    else:
        return False, 'name', f'`{name}` not found'


def update_task():
    pass


def list_tasks() -> list:
    files = os.listdir('tasks')
    files = [f for f in files if not re.match(r'__.*__', f)]
    files = [f[:-5] for f in files if f.endswith('.json')]

    tasks = []

    for t in files:
        with open(f'tasks/{t}.json', 'r') as f:
            args = json.load(f)

        task = load_task(t)
        loc = task['data_locator'](index_path='data/ind.csv', data_root='data/test', random_state=task['random_state'])
        data_example = loc.next().tail(5).to_csv()

        tasks.append({
            'name': t,
            'args': args,
            'date': datetime.fromtimestamp(os.path.getctime(f'tasks/{t}.json')).strftime("%Y-%m-%d %H:%M:%S"),
            'data_example': data_example
        })

    return tasks


def export_task(name: str) -> dict:
    if os.path.exists(f'tasks/{name}.json'):
        with open(f'tasks/{name}.json', 'r') as f:
            return json.load(f)
    else:
        return {}


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
