from deprecated.environment.trade_new import Env, TensorboardCallback
from torch.utils.tensorboard import SummaryWriter


class Pipeline:
    def __init__(self, tasks: list):
        self.tasks = tasks
        self._check_components()  # check components

    def _check_components(self):
        for p, i in zip(self.tasks, range(len(self.tasks))):
            assert 'name' in p, f'pipeline[{i}]: name is required'
            # algorithm
            assert 'algorithm' in p, f'pipeline[{i}]: algorithm is required'
            assert 'algorithm_args' in p, f'pipeline[{i}]: algorithm_args is required'
            assert 'learn_args' in p, f'pipeline[{i}]: learn_args is required'
            # environment
            assert 'helper' in p, f'pipeline[{i}]: helper is required'
            helper = p['helper']()
            if helper.observation_space is None or helper.action_space is None:
                raise ValueError(f'pipeline[{i}]: nither observation_space nor action_space can be None')
            del helper

    def _train(self, task: dict):
        helper = task['helper']()
        env = Env(train=True,
                  data_getter=helper.data_getter,
                  data_preprocess=helper.data_preprocess,
                  action_decoder=helper.action_decoder,
                  observation_space=helper.observation_space,
                  action_space=helper.action_space)
        model = task['algorithm'](**task['algorithm_args'], env=env, verbose=1,
                                  tensorboard_log=f'tensorboard/{task["name"]}')
        model.learn(**task['learn_args'], progress_bar=True, callback=TensorboardCallback())
        model.save(f'model/{task["name"]}')

        # GC
        del env
        del model

    def _test(self, task: dict):
        helper = task['helper']()
        writer = SummaryWriter(f'tensorboard/{task["name"]}_test')
        env = Env(train=False,
                  data_getter=helper.data_getter,
                  data_preprocess=helper.data_preprocess,
                  action_decoder=helper.action_decoder,
                  observation_space=helper.observation_space,
                  action_space=helper.action_space)
        model = task['algorithm'].load(f'model/{task["name"]}')
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
                # writer = SummaryWriter(f'tensorboard/{task["name"]}_test')

        # GC
        writer.close()
        del env
        del model

    def train(self):
        yield  # ready for first
        for task in self.tasks:
            self._train(task)
            yield task['name']  # ready for next

        yield None

    def test(self):
        yield  # ready for first
        for task in self.tasks:
            self._test(task)
            yield task['name']  # ready for next

        yield None


if __name__ == '__main__':
    raise NotImplementedError('this file not meant to be executed')
