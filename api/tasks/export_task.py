from .load_task import load_task
from typing import Tuple, Union


def export_task(name: str) -> Tuple[bool, str, Union[str, dict]]:
    task = load_task(name, _eval=False)
    if task is None:
        return False, 'name', f'`{name}` not found'

    return True, 'success', task


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
