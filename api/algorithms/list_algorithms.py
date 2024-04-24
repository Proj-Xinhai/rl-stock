import importlib


def list_algorithms():
    sb3 = importlib.import_module('stable_baselines3').__dict__
    sb3_contrib = importlib.import_module('sb3_contrib').__dict__

    algorithms = []

    for k, v in sb3.items():
        # check if is a class
        if isinstance(v, type) and 'policy' in v.__init__.__code__.co_varnames:
            algorithms.append({
                'name': k,
                'description': v.__doc__,
                'args': v.__init__.__code__.co_varnames
            })

    for k, v in sb3_contrib.items():
        if isinstance(v, type) and 'policy' in v.__init__.__code__.co_varnames:
            algorithms.append({
                'name': k,
                'description': v.__doc__,
                'args': v.__init__.__code__.co_varnames
            })

    return algorithms


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
