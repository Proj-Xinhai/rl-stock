import json


def set_work(uuid: str, args: dict):
    with open(f'tasks/works/{uuid}.json', 'w') as f:
        json.dump(args, f)


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')