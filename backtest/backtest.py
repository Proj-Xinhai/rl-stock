from typing import Tuple, Optional
from api.works import load_work


def backtest(stock: str, start: str, end: str, model: str):
    work = load_work(model)


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
