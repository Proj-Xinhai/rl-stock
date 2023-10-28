import json
from uuid import uuid4
from time import time
from typing import Tuple


def create_work(task_name: str, num_work: int) -> Tuple[bool, str, str]:
    for _ in range(num_work):
        uuid = uuid4().hex

        with open(f'tasks/works/{uuid}.json', 'w') as f:
            args = {
                'id': uuid,
                'task_name': task_name,
                'status': 0,  # 0: pending, 1: running, 2: finished, -1: error
                'detail': '',
                'timeline': [{
                    'name': 'create',
                    'from': time(),
                    'to': time(),
                    'status': 2,
                    'detail': ''
                }, {
                    'name': 'train',
                    'from': 0,
                    'to': 0,
                    'status': 0,
                    'detail': ''
                }, {
                    'name': 'test',
                    'from': 0,
                    'to': 0,
                    'status': 0,
                    'detail': ''
                }],
                'evaluation': []
            }
            json.dump(args, f)

    return True, 'success', ''


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
