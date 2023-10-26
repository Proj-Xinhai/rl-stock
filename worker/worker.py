from worker.train import train
from worker.test import test
from util.load_task import load_task
from api.works import set_work_timeline
import json


def worker(uuid: str):
    with open(f'tasks/works/{uuid}.json', 'r') as f:
        args = json.load(f)

    task = load_task(args['task_name'])
    algorithm = task['algorithm']
    algorithm_args = task['algorithm_args']
    learn_args = task['learn_args']
    data_locator = task['data_locator']
    environment = task['environment']
    random_state = task['random_state']

    set_work_timeline(uuid, 'train', 1, '')
    try:
        model = train(uuid, data_locator, environment, algorithm, algorithm_args, learn_args, random_state)
    except KeyboardInterrupt:
        set_work_timeline(uuid, 'train', -1, 'KeyboardInterrupt')
        return False, 'failed', 'KeyboardInterrupt'
    except Exception as e:
        set_work_timeline(uuid, 'train', -1, str(e))
        return False, 'failed', str(e)
    set_work_timeline(uuid, 'train', 2, '')

    set_work_timeline(uuid, 'test', 1, '')
    try:
        test(uuid, model, data_locator, environment, random_state)
    except KeyboardInterrupt:
        set_work_timeline(uuid, 'test', -1, 'KeyboardInterrupt')
        return False, 'failed', 'KeyboardInterrupt'
    except Exception as e:
        set_work_timeline(uuid, 'test', -1, str(e))
        return False, 'failed', str(e)
    set_work_timeline(uuid, 'test', 2, '')

    return True, 'success', ''
