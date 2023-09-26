import pandas as pd
from typing import Any, Tuple


class BasicPipelineHelper:
    def __init__(self):
        self.observation_space = None
        self.action_space = None

    def data_getter(self, stock_num: str, is_train: bool = True) -> pd.DataFrame:
        raise NotImplementedError('data_getter not implemented')

    def data_preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        raise NotImplementedError('data_preprocess not implemented')

    def action_decoder(self, action: Any) -> Any:
        raise NotImplementedError('action_decoder not implemented')


if __name__ == '__main__':
    raise NotImplementedError('this file not meant to be executed')
