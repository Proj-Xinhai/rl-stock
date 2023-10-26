from .load_task import load_task
import os
import re
from datetime import datetime


def list_tasks() -> list:
    files = os.listdir('tasks')
    files = [f for f in files if not re.match(r'__.*__', f)]
    files = [f[:-5] for f in files if f.endswith('.json')]

    tasks = []

    for t in files:
        args = load_task(t, _eval=False)
        if args is None:
            continue

        task = load_task(t)
        loc = task['data_locator'](index_path='data/ind.csv', data_root='data/test', random_state=task['random_state'])
        data_example = loc.next().tail(5).to_csv()

        tasks.append({
            'name': t,
            'args': args,
            'date': datetime.fromtimestamp(os.path.getctime(f'tasks/{t}.json')).strftime("%Y-%m-%d %H:%M:%S"),
            'data_example': data_example
        })

    return tasks


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
