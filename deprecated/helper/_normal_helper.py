import numpy as np
import pandas as pd
import gymnasium.spaces as spaces
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Any
from deprecated.pipeline_helper import BasicPipelineHelper


class NormalHelper(BasicPipelineHelper):
    def __init__(self):
        super(NormalHelper, self).__init__()  # 繼承父類別的__init__()
        self.observation_space = spaces.Box(
            low=np.array([np.concatenate([np.zeros(7), -np.ones(7), np.array([-np.inf])])]),
            high=np.array([np.concatenate([np.ones(7), np.ones(7), np.array([np.inf])])]),
            shape=(1, 7 + 7 + 1),  # 7: 個股資料, 4: SMA, 7: 法人買賣超, 1: 持有量
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def data_getter(self, stock_num: str, is_train: bool = True) -> pd.DataFrame:
        data_path = 'data/train' if is_train else 'data/test'  # 資料路徑

        # 個股資料
        s = pd.read_csv(f'{data_path}/個股/{stock_num}.csv').reset_index(drop=True).set_index('Date')
        # 三大法人買賣超
        i = pd.read_csv(f'{data_path}/法人買賣超日報_個股/{stock_num}.csv', index_col=0).reset_index(
            drop=True).set_index('Date')

        # 正規化法人買賣超，且只保留買賣超部分
        i = i.apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        i = i[['外陸資買賣超股數(不含外資自營商)', '外資自營商買賣超股數', '投信買賣超股數', '自營商買賣超股數',
               '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)', '三大法人買賣超股數']]
        i = i.apply(lambda x: x.apply(lambda y: 1 if float(y) > 0 else -1 if float(y) < 0 else 0))
        # i = i.shift(1) # 往下偏移一天 (因為當天結束才會統計買賣超資訊，實務上交易日當天是不知道當天買賣超資訊的)
        # i = i.fillna(0)

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        data = pd.concat([s, i], axis=1, sort=True).dropna()

        return data

    def data_preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        scaler = MinMaxScaler()
        scaler.fit(data['Close'].values.reshape(-1, 1))
        data['Open'] = scaler.transform(data['Open'].values.reshape(-1, 1)).reshape(-1)
        data['High'] = scaler.transform(data['High'].values.reshape(-1, 1)).reshape(-1)
        data['Low'] = scaler.transform(data['Low'].values.reshape(-1, 1)).reshape(-1)
        data['Close'] = scaler.transform(data['Close'].values.reshape(-1, 1)).reshape(-1)

        return data, scaler

    def action_decoder(self, action: Any) -> Any:
        return bool(action) if action != 2 else None


EXPORT = NormalHelper
DESCRIPT = 'NormalHelper with only institutional investors'
