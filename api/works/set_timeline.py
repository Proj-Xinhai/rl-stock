from .load_work import load_work
from .set_work import set_work
from time import time
from typing import Tuple


def set_timeline(uuid: str, name: str, status: int, detail: str) -> Tuple[bool, str, str]:
    work = load_work(uuid)
    if work is None:
        return False, 'work', 'work not found'

    for t in work['timeline']:
        if t['name'] == name:
            if status == 1:
                t['from'] = time()
            elif status == 2 or status == -1:
                t['to'] = time()
            t['status'] = status
            t['detail'] = detail

    set_work(uuid, work)

    return True, 'success', ''


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
