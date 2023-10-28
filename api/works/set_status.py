from .load_work import load_work
from .set_work import set_work
from typing import Tuple


def set_status(uuid: str, status: int, detail: str) -> Tuple[bool, str, str]:
    work = load_work(uuid)
    if work is None:
        return False, 'work', 'work not found'

    work['status'] = status
    work['detail'] = detail

    set_work(uuid, work)

    return True, 'success', ''


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
