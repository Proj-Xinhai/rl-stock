import json
import os
from typing import Optional


def load_work(uuid: str) -> Optional[dict]:
    if not os.path.exists(f'tasks/works/{uuid}.json'):
        return None

    with open(f'tasks/works/{uuid}.json', 'r') as f:
        args = json.load(f)

    return args


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
