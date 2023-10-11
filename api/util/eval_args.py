from ast import literal_eval
from typeguard import check_type, TypeCheckError
from typing import Tuple, Union

from api.util.get_algorithm import get_algorithm


def eval_args(algorithm: str, algorithm_args: dict, learn_args: dict) -> Union[(
        Tuple[None, None, str, str], Tuple[dict, dict, None, None])]:
    sb3 = get_algorithm(algorithm)
    if sb3 is None:
        return None, None, 'algorithm', 'algorithm not found'

    for k, v in algorithm_args.items():
        try:
            v = literal_eval(v)
        except ValueError:
            pass  # if not literal, pass. consider it as string

        if k not in sb3.__init__.__code__.co_varnames:
            return None, None, 'algorithm_args', f'`{k}` not found'

        try:
            check_type(v, sb3.__init__.__annotations__[k])
        except TypeCheckError:
            return None, None, 'algorithm_args', f'`{k} ({v})` type invalid'

        algorithm_args[k] = v

    for k, v in learn_args.items():
        try:
            v = literal_eval(v)
        except ValueError:
            pass

        if k not in sb3.learn.__code__.co_varnames:
            return None, None, 'learn_args', f'`{k}` not found'

        try:
            check_type(v, sb3.learn.__annotations__[k])
        except TypeCheckError:
            return None, None, 'learn_args', f'`{k} ({v})` type invalid'

        learn_args[k] = v

    return algorithm_args, learn_args, None, None


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
