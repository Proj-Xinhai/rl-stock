import os
import re
import json
from datetime import datetime
from typing import Tuple, Optional

from api.util.load_task import load_task
from api.list_helper import list_helper
from api.util.eval_args import eval_args


def create_task(name: str, algorithm: str, algorithm_args: dict, learn_args: dict, helper: str) -> Tuple[bool, str, str]:
    # check if name is valid
    if name == '':
        return False, 'name', f'cannot be empty'

    _, _, error, detail = eval_args(algorithm, algorithm_args, learn_args)

    if error is not None:
        return False, error, detail

    # check if helper is valid (in general, this will not be invalid)
    if next((h for h in list_helper() if h['name'] == helper), None) is None:
        return False, 'helper', f'`{helper}` not found'

    args = {
        'name': name,  # str
        'algorithm': algorithm,  # class name like A2C
        'algorithm_args': algorithm_args,  # dict
        'learn_args': learn_args,  # dict
        'helper': helper  # .py file
    }

    # first check if task already exists, if so, add number
    if os.path.exists(f'tasks/{name}.json'):
        i = 1
        while os.path.exists(f'tasks/{name}_{i}.json'):
            i += 1
        name = f'{name}_{i}'
        args['name'] = name

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

        task = load_task(t)['helper']()
        data_example = task.data_getter('2330')
        preprocess_example, _ = task.data_preprocess(task.data_getter('2330'))

        tasks.append({
            'name': t,
            'args': args,
            'date': datetime.fromtimestamp(os.path.getctime(f'tasks/{t}.json')).strftime("%Y-%m-%d %H:%M:%S"),
            'data_example': data_example.head(5).to_csv(),
            'preprocess_example': preprocess_example.head(5).to_csv()
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
