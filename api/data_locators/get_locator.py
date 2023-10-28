import os
import importlib
from typing import Optional


def get_locator(name: str) -> Optional[dict]:
    if not os.path.exists(f'data_locator/{name}.py'):
        return None

    return importlib.import_module('data_locator.' + name).__dict__['EXPORT']


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
