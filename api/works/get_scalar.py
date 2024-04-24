import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def get_scalar(uuid: str, gets=None) -> dict:

    if gets is None:
        gets = ['train', 'test']

    data = {
        'train': [],
        'test': []
    }

    try:
        if 'train' not in gets:
            raise FileNotFoundError('train not in gets')

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
    except FileNotFoundError:
        data['train'] = []

    try:
        if 'test' not in gets:
            raise FileNotFoundError('test not in gets')

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
    except FileNotFoundError:
        data['test'] = []

    return data


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
