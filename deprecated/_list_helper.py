import os
import re
import importlib


def list_helper():
    files = os.listdir('util/helper')
    files = [f for f in files if not re.match(r'__.*__', f)]
    files = [f[:-3] for f in files if f.endswith('.py') and not f.startswith('_')]

    helpers = []

    for h in files:
        loaded_helper = importlib.import_module('util.helper.' + h).__dict__
        helpers.append({
            'name': h,
            'description': loaded_helper['DESCRIPT']
        })

    return helpers


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
