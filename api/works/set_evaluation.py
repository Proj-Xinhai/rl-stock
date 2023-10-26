from .load_work import load_work
from .set_work import set_work


def set_evaluation(uuid: str, evaluation: dict):
    work = load_work(uuid)
    if work is None:
        return False, 'work', 'work not found'

    work['evaluation'].append(evaluation)

    set_work(uuid, work)

    return True, 'success', ''


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
