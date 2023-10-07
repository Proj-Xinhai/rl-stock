import os
import re
import json
from datetime import datetime
from uuid import uuid4
from time import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def create_work(task_name: str, num_work: int):
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
            }
            json.dump(args, f)
    return True, 'success', ''


def set_work_timeline(uuid: str, name: str, status: int, detail: str):
    with open(f'tasks/works/{uuid}.json', 'r') as f:
        args = json.load(f)

    for t in args['timeline']:
        if t['name'] == name:
            if status == 1:
                t['from'] = time()
            elif status == 2:
                t['to'] = time()
            t['status'] = status
            t['detail'] = detail

    with open(f'tasks/works/{uuid}.json', 'w') as f:
        json.dump(args, f)

    return True, 'success', ''


def set_work_status(uuid: str, status: int, detail: str):
    with open(f'tasks/works/{uuid}.json', 'r') as f:
        args = json.load(f)

    args['status'] = status
    args['detail'] = detail

    with open(f'tasks/works/{uuid}.json', 'w') as f:
        json.dump(args, f)

    return True, 'success', ''


def remove_work():
    pass


def update_work():
    pass


def list_works():
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
            'date': datetime.fromtimestamp(os.path.getctime(f'tasks/works/{t}.json')).strftime("%Y-%m-%d %H:%M:%S"),
        })

    return works


def get_scaler(uuid: str):
    data = {
        'train': [],
        'test': []
    }

    if os.path.exists(f'tasks/works/{uuid}/{uuid}_1'):
        train = EventAccumulator(f'tasks/works/{uuid}/{uuid}_1')
        train.Reload()

        train_temp = pd.DataFrame()

        for tag in train.Tags()['scalars']:
            df = pd.DataFrame(train.Scalars(tag))
            train_temp = pd.concat([train_temp, pd.DataFrame(
                {
                    'tag': tag,
                    'step': df['step'],
                    'value': df['value']
                }
            )])

        train_temp = train_temp.groupby(['tag'])[['step', 'value']].agg({
            'step': lambda x: list(x),
            'value': lambda x: list(x)
        }).reset_index()

        train_temp['group'] = [t.split('/')[0] for t in train_temp['tag']]

        train_temp = train_temp.groupby(['group'])[['tag', 'step', 'value']].apply(lambda x: x.to_dict('records'))
        train_temp = train_temp.reset_index(name='data').to_dict('records')

        data['train'] = train_temp

    if os.path.exists(f'tasks/works/{uuid}/{uuid}_test'):
        test = EventAccumulator(f'tasks/works/{uuid}/{uuid}_test')
        test.Reload()

        test_temp = pd.DataFrame()

        for tag in test.Tags()['scalars']:
            df = pd.DataFrame(test.Scalars(tag))
            test_temp = pd.concat([test_temp, pd.DataFrame(
                {
                    'tag': tag,
                    'step': df['step'],
                    'value': df['value']
                }
            )])

        test_temp = test_temp.groupby(['tag'])[['step', 'value']].agg({
            'step': lambda x: list(x),
            'value': lambda x: list(x)
        }).reset_index()

        test_temp['group'] = [t.split('/')[0] for t in test_temp['tag']]

        test_temp = test_temp.groupby(['group'])[['tag', 'step', 'value']].apply(lambda x: x.to_dict('records'))
        test_temp = test_temp.reset_index(name='data').to_dict('records')

        data['test'] = test_temp

    return data


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
