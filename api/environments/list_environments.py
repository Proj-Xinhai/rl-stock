import os
import re
import importlib


def list_environments() -> list:
    files = os.listdir('environment')
    files = [f for f in files if not re.match(r'__.*__', f)]
    files = [f[:-3] for f in files if f.endswith('.py') and not f.startswith('_')]

    environments = []

    for loc in files:
        loaded_environment = importlib.import_module('environment.' + loc).__dict__
        environments.append({
            'name': loc,
            'description': loaded_environment['DESCRIPT']
        })

    return environments


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
