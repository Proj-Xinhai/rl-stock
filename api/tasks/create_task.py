from ..algorithms import eval_args, get_algorithm
from ..data_locators import list_locators
from ..environments import get_environment
from typing import Optional, Tuple
import os
from pathvalidate import sanitize_filename
import json
import importlib


def create_task(name: str,
                algorithm: str,
                algorithm_args: dict,
                learn_args: dict,
                data_locator: str,
                environment: str,
                random_state: Optional[int]) -> Tuple[bool, str, str]:
    # check if name is valid
    if name == '':
        return False, 'name', f'cannot be empty'

    _, _, error, detail = eval_args(algorithm, algorithm_args, learn_args)

    if error is not None:
        return False, error, detail

    # check if data locator is valid (in general, this will not be invalid)
    if next((loc for loc in list_locators() if loc['name'] == data_locator), None) is None:
        return False, 'data_locator', f'`{data_locator}` not found'

    # check if environment is valid (in general, this will not be invalid)
    if not os.path.exists(f'environment/{environment}.py'):
        return False, 'environment', f'`{environment}` not found'

    args = {
        'name': name,  # str
        'algorithm': algorithm,  # class name like A2C
        'algorithm_args': algorithm_args,  # dict
        'learn_args': learn_args,  # dict
        'data_locator': data_locator,  # .py file
        'environment': environment,  # .py file
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
            env = get_environment(environment)[0](data_locator=loc)
            sb3(**algorithm_args, env=env)
    except Exception as e:
        return False, 'algorithm_args', str(e)

    # write to file
    with open(f'tasks/{name}.json', 'w') as f:
        json.dump(args, f)

    return True, 'success', name


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
