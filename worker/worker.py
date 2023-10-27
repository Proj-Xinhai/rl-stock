from .train import train
from .test import test
from api.tasks import load_task
from api.works import load_work, set_timeline


def worker(uuid: str):
    work = load_work(uuid)

    task = load_task(work['task_name'])
    algorithm = task['algorithm']
    algorithm_args = task['algorithm_args']
    learn_args = task['learn_args']
    data_locator = task['data_locator']
    environment = task['environment']
    random_state = task['random_state']

    set_timeline(uuid, 'train', 1, '')
    try:
        model = train(uuid, data_locator, environment, algorithm, algorithm_args, learn_args, random_state)
    except KeyboardInterrupt:
        set_timeline(uuid, 'train', -1, 'KeyboardInterrupt')
        return False, 'failed', 'KeyboardInterrupt'
    except Exception as e:
        set_timeline(uuid, 'train', -1, str(e))
        return False, 'failed', str(e)
    set_timeline(uuid, 'train', 2, '')

    set_timeline(uuid, 'test', 1, '')
    try:
        test(uuid, model, data_locator, environment, random_state, is_recurrent=algorithm.__name__ == 'RecurrentPPO')
    except KeyboardInterrupt:
        set_timeline(uuid, 'test', -1, 'KeyboardInterrupt')
        return False, 'failed', 'KeyboardInterrupt'
    except Exception as e:
        set_timeline(uuid, 'test', -1, str(e))
        return False, 'failed', str(e)
    set_timeline(uuid, 'test', 2, '')

    return True, 'success', ''
