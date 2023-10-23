import io
import os
import re
import json
from datetime import datetime
from uuid import uuid4
from time import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import zipfile


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
                'evaluation': []
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
            elif status == 2 or status == -1:
                t['to'] = time()
            t['status'] = status
            t['detail'] = detail

    # set_work_status(uuid, status, detail)

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
            'evaluation': args['evaluation'],
            'date': datetime.fromtimestamp(os.path.getctime(f'tasks/works/{t}.json')).strftime("%Y-%m-%d %H:%M:%S"),
        })

    return sorted(works, key=lambda x: x['date'], reverse=False)


def get_scalar(uuid: str, gets=None):

    if gets is None:
        gets = ['train', 'test']

    data = {
        'train': [],
        'test': []
    }

    try:
        if 'train' not in gets:
            raise Exception('train not in gets')

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
                )]).ffill()

            train_temp = train_temp.groupby(['tag'])[['step', 'value']].agg({
                'step': lambda x: tuple(x),  # to make it can be set as index
                'value': lambda x: tuple(x)
            }).reset_index()

            train_temp['group'] = [t.split('/')[0] for t in train_temp['tag']]

            train_temp = train_temp.groupby(['group', 'step'])[['tag', 'value']].apply(lambda x: x.to_dict('records'))
            train_temp = train_temp.reset_index(name='data')

            data['train'] = train_temp.to_dict('records')
    except Exception as e:
        data['train'] = []

    try:
        if 'test' not in gets:
            raise Exception('test not in gets')

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
                )]).ffill()

            test_temp = test_temp.groupby(['tag'])[['step', 'value']].agg({
                'step': lambda x: tuple(x),  # to make it can be set as index
                'value': lambda x: tuple(x)
            }).reset_index()

            test_temp['group'] = [t.split('/')[0] for t in test_temp['tag']]

            test_temp = test_temp.groupby(['group', 'step'])[['tag', 'value']].apply(lambda x: x.to_dict('records'))
            test_temp = test_temp.reset_index(name='data')

            data['test'] = test_temp.to_dict('records')
    except Exception as e:
        data['test'] = []

    return data


def set_evaluation(uuid: str, evaluation: dict):
    with open(f'tasks/works/{uuid}.json', 'r') as f:
        args = json.load(f)

    args['evaluation'].append(evaluation)

    with open(f'tasks/works/{uuid}.json', 'w') as f:
        json.dump(args, f)

    return True, 'success', ''


def export_work(uuid: str):
    if os.path.exists(f'tasks/works/{uuid}.json'):
        with open(f'tasks/works/{uuid}.json', 'r') as f:
            f = json.load(f)
            if f['status'] != 2 or not os.path.exists(f'tasks/works/{uuid}'):
                return False, 'status', 'work is failed or not yet finished'

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'a', zipfile.ZIP_DEFLATED, False) as zf:
            zf.write(f'tasks/works/{uuid}.json', arcname=f'{uuid}.json')
            for root, dirs, files in os.walk(f'tasks/works/{uuid}'):
                for file in files:
                    zf.write(os.path.join(root, file), arcname=os.path.join(root, file)[len(f'tasks/works') + 1:])

        return True, 'success', buffer.getvalue()

    else:
        return False, 'uuid', f'`{uuid}` not found'


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
