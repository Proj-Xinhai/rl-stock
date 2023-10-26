from ..algorithms import eval_args, get_algorithm
from ..data_locators import get_locator
import os
import json
from typing import Optional


def load_task(name: str, _eval: bool = True) -> Optional[dict]:
    if not os.path.exists(f'tasks/{name}.json'):
        return None

    with open(f'tasks/{name}.json', 'r') as f:
        args = json.load(f)

    if not _eval:
        return args

    args['algorithm_args'], args['learn_args'], error, detail = \
        eval_args(args['algorithm'], args['algorithm_args'], args['learn_args'])

    if error is not None:
        raise RuntimeError(f'Error in task `{name}`: {error} ({detail})')

    args['algorithm'] = get_algorithm(args['algorithm'])
    args['data_locator'] = get_locator(args['data_locator'])
    args['random_state'] = None if args['random_state'] == '' else args['random_state']

    return args


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
