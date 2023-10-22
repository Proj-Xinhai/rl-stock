import os
import re
import importlib


def list_data_locator():
    files = os.listdir('util/data_locator')
    files = [f for f in files if not re.match(r'__.*__', f)]
    files = [f[:-3] for f in files if f.endswith('.py') and not f.startswith('_')]

    locators = []

    for loc in files:
        loaded_data_locator = importlib.import_module('util.data_locator.' + loc).__dict__
        locators.append({
            'name': loc,
            'description': loaded_data_locator['DESCRIPT']
        })

    return locators


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
