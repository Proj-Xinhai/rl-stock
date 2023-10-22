import json
from torch.utils.tensorboard import SummaryWriter
from environment.trade_enhance import Env, TensorboardCallback

from api.util.load_task import load_task
from api.works import set_work_timeline, set_evaluation


def run_work(uuid: str):
    with open(f'tasks/works/{uuid}.json', 'r') as f:
        args = json.load(f)

    task = load_task(args['task_name'])

    algorithm = task['algorithm']
    algorithm_args = task['algorithm_args']
    learn_args = task['learn_args']
    data_locator = task['data_locator']

    try:
        env = Env(data_locater=data_locator)

        set_work_timeline(uuid, 'train', 1, '')

        # train
        model = algorithm(**algorithm_args, env=env, verbose=1, tensorboard_log=f'tasks/works/{uuid}')
        model.learn(**learn_args, progress_bar=True, callback=TensorboardCallback(), tb_log_name=f'{uuid}')
        model.save(f'tasks/works/{uuid}/{uuid}')

        set_work_timeline(uuid, 'train', 2, '')

        del env
    except Exception as e:
        set_work_timeline(uuid, 'train', -1, str(e))
        return False, 'failed', str(e)

    try:
        # test
        env = Env(data_locater=data_locator, data_root='data/test')

        writer = SummaryWriter(f'tasks/works/{uuid}/{uuid}_test')

        set_work_timeline(uuid, 'test', 1, '')

        obs, info = env.reset()
        step_count = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)

            writer.add_scalar('env/hold', info['hold'], step_count)
            writer.add_scalar('env/holding_count', info['holding_count'], step_count)
            writer.add_scalar('env/net', info['net'], step_count)
            writer.add_scalar('env/net_exclude_settlement', info['net_exclude_settlement'], step_count)
            writer.add_scalar('env/reward', rewards, step_count)

            step_count += 1

            if terminated:
                if info['finish']:
                    break
                obs, info = env.reset()

        set_work_timeline(uuid, 'test', 2, '')

        try:
            set_evaluation(uuid, {
                'name': 'roi_net',
                'value': info['net'] / info['cost']
            })
            set_evaluation(uuid, {
                'name': 'roi_net_exclude_settlement',
                'value': info['net_exclude_settlement'] / info['cost']
            })
        except ZeroDivisionError:
            set_evaluation(uuid, {
                'name': 'zero_division_error',
                'value': -1
            })

        writer.close()
    except Exception as e:
        set_work_timeline(uuid, 'test', -1, str(e))
        return False, 'failed', str(e)

    return True, 'success', ''


if __name__ == '__main__':
    raise RuntimeError('This file is not meant to be executed')
