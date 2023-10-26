import os
import re
import json
from datetime import datetime


def list_works() -> list:
    files = os.listdir('tasks/works')
    files = [f for f in files if not re.match(r'__.*__', f)]
    files = [f[:-5] for f in files if f.endswith('.json')]

    works = []

    for t in files:
        with open(f'tasks/works/{t}.json', 'r') as f:
            args = json.load(f)

        works.append({
            'id': t,
            'task_name': args['task_name'],
            'status': args['status'],
            'detail': args['detail'],
            'timeline': args['timeline'],
            'evaluation': args['evaluation'],
            'date': datetime.fromtimestamp(os.path.getctime(f'tasks/works/{t}.json')).strftime("%Y-%m-%d %H:%M:%S"),
        })

    return sorted(works, key=lambda x: x['date'], reverse=False)


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
