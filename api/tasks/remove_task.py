import os
from typing import Tuple


def remove_task(name: str) -> Tuple[bool, str, str]:
    # TODO: remove task must remove works too
    if os.path.exists(f'tasks/{name}.json'):
        os.remove(f'tasks/{name}.json')
        return True, 'success', name
    else:
        return False, 'name', f'`{name}` not found'


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
