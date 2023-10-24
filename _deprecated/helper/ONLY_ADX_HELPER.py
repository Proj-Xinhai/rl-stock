import numpy as np
import pandas as pd
import gymnasium.spaces as spaces
import talib
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Any
from deprecated.pipeline_helper import BasicPipelineHelper


class OnlyADXHelper(BasicPipelineHelper):
    def __init__(self):
        super(OnlyADXHelper, self).__init__()  # 繼承父類別的__init__()
        self.observation_space = spaces.Box(
            low=np.array([np.concatenate([np.zeros(1), np.zeros(1), np.array([-np.inf])])]),
            high=np.array([np.concatenate([np.ones(1), np.ones(1), np.array([np.inf])])]),
            shape=(1, 1 + 1 + 1),  # 1 收盤價, 1: ADX, 1: 持有量
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def data_getter(self, stock_num: str, is_train: bool = True) -> pd.DataFrame:
        data_path = 'data/train' if is_train else 'data/test'  # 資料路徑

        # 個股資料
        s = pd.read_csv(f'{data_path}/個股/{stock_num}.csv').reset_index(drop=True).set_index('Date')
        # s = s[['Close']]  # 只保留收盤價  # 先保留所有欄位供後續計算技術指標使用

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        data = pd.concat([s], axis=1, sort=True).dropna()

        # 先合併完，再算技術指標 (避免dropna把開頭的資料刪掉)
        # ADX
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

        # 只保留收盤價與 ADX
        data = data[['Close', 'ADX']]

        """
        補空值
        作法: 直接補0
        """
        data = data.fillna(0)

        return data

    def data_preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        scaler = MinMaxScaler()
        scaler.fit(data['Close'].values.reshape(-1, 1))
        # 收盤價
        data['Close'] = scaler.transform(data['Close'].values.reshape(-1, 1)).reshape(-1)

        # ADX
        data['ADX'] = data['ADX'] / 100  # ADX 本身是 0~100 的數值，除以 100 使其變成 0~1

        return data, scaler

    def action_decoder(self, action: Any) -> Any:
        return bool(action) if action != 2 else None


EXPORT = OnlyADXHelper
DESCRIPT = 'adx data only'
