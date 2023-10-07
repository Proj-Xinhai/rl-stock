import importlib
import json

from api.util.get_algorithm import get_algorithm
from api.util.eval_args import eval_args


def load_task(name: str):
    with open(f'tasks/{name}.json', 'r') as f:
        args = json.load(f)

    args['algorithm_args'], args['learn_args'], error, detail = eval_args(args['algorithm'], args['algorithm_args'], args['learn_args'])

    if error is not None:
        raise RuntimeError(f'Error in task `{name}`: {error} ({detail})')

    args['algorithm'] = get_algorithm(args['algorithm'])
    args['helper'] = importlib.import_module('util.helper.' + args['helper']).__dict__['EXPORT']

    return args


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
