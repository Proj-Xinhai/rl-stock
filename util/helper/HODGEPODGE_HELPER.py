import numpy as np
import pandas as pd
import gymnasium.spaces as spaces
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Any
from util.pipeline_helper import BasicPipelineHelper
import talib


class HodgepodgeHelper(BasicPipelineHelper):
    def __init__(self):
        super(HodgepodgeHelper, self).__init__()  # 繼承父類別的__init__()
        self.observation_space = spaces.Box(
            low=np.array([np.concatenate([
                np.zeros(1),  # 收盤價
                -np.ones(7),  # 法人買賣超
                np.zeros(4),  # SMA
                np.array([-np.inf] * 3),  # MACD
                np.zeros(1),  # RSI
                np.array([-np.inf]),  # CCI
                np.zeros(1),  # ADX
                np.array([-np.inf])  # 持有量
            ])]),
            high=np.array([np.concatenate([
                np.ones(1),  # 收盤價
                np.ones(7),  # 法人買賣超
                np.ones(4),  # SMA
                np.array([np.inf] * 3),  # MACD
                np.ones(1),  # RSI
                np.array([np.inf]),  # CCI
                np.ones(1),  # ADX
                np.array([np.inf])  # 持有量
            ])]),
            shape=(1, 1 + 7 + 4 + 3 + 1 + 1 + 1 + 1 + 1),
            # 1 收盤價, 7: 法人買賣超, 4: SMA, 3: MACD, 1: RSI, 1: CCI, 1: ADX, 1: 持有量
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def data_getter(self, stock_num: str, is_train: bool = True) -> pd.DataFrame:
        data_path = 'data/train' if is_train else 'data/test'  # 資料路徑

        # 個股資料
        s = pd.read_csv(f'{data_path}/個股/{stock_num}.csv').reset_index(drop=True).set_index('Date')
        # s = s[['Close']]  # 只保留收盤價
        # 三大法人買賣超
        i = pd.read_csv(f'{data_path}/法人買賣超日報_個股/{stock_num}.csv', index_col=0).reset_index(
            drop=True).set_index('Date')

        # 正規化法人買賣超，且只保留買賣超部分
        i = i.apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        i = i[['外陸資買賣超股數(不含外資自營商)', '外資自營商買賣超股數', '投信買賣超股數', '自營商買賣超股數',
               '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)', '三大法人買賣超股數']]
        i = i.apply(lambda x: x.apply(lambda y: 1 if float(y) > 0 else -1 if float(y) < 0 else 0))
        i = i.shift(1)  # 往下偏移一天 (因為當天結束才會統計買賣超資訊，實務上交易日當天是不知道當天買賣超資訊的)
        i = i.fillna(0)  # 用0補空值 (影響應該不會太大)

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        data = pd.concat([s, i], axis=1, sort=True).dropna()

        # 先合併完，再算技術指標 (避免dropna把開頭的資料刪掉)
        # SMA
        data['MA5'] = talib.MA(data['Close'], timeperiod=5)
        data['MA10'] = talib.MA(data['Close'], timeperiod=10)
        data['MA20'] = talib.MA(data['Close'], timeperiod=20)
        data['MA60'] = talib.MA(data['Close'], timeperiod=60)
        # MACD
        data['MACD'], data['MACDsignal'], data['MACDhist'] = (
            talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9))
        # RSI
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        # CCI
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
        # ADX
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

        data = data[['Close',  # 收盤價
                     '外陸資買賣超股數(不含外資自營商)', '外資自營商買賣超股數', '投信買賣超股數', '自營商買賣超股數',
                     '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)', '三大法人買賣超股數',  # 法人買賣超
                     'MA5', 'MA10', 'MA20', 'MA60',  # SMA
                     'MACD', 'MACDsignal', 'MACDhist',  # MACD
                     'RSI',  # RSI
                     'CCI',  # CCI
                     'ADX'  # ADX
                     ]]

        return data

    def data_preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        scaler = MinMaxScaler()
        scaler.fit(data['Close'].values.reshape(-1, 1))
        # 收盤價
        data['Close'] = scaler.transform(data['Close'].values.reshape(-1, 1)).reshape(-1)
        # 法人買賣超
        # 已經經過正規化
        # SMA
        data['MA5'] = scaler.transform(data['MA5'].values.reshape(-1, 1)).reshape(-1)
        data['MA10'] = scaler.transform(data['MA10'].values.reshape(-1, 1)).reshape(-1)
        data['MA20'] = scaler.transform(data['MA20'].values.reshape(-1, 1)).reshape(-1)
        data['MA60'] = scaler.transform(data['MA60'].values.reshape(-1, 1)).reshape(-1)
        # MACD
        # 難以正規化，直接送出原始值
        # RSI
        data['RSI'] = data['RSI'] / 100  # RSI 本身是 0~100 的數值，除以 100 使其變成 0~1
        # CCI
        data['CCI'] = data['CCI'] / 100  # CCI 本身無上下限，通常落在 -100 ~ 100 之間，除以 100 使其變成 -1 ~ 1 （有些值會超出這個範圍）
        # ADX
        data['ADX'] = data['ADX'] / 100  # ADX 本身是 0~100 的數值，除以 100 使其變成 0~1

        return data, scaler

    def action_decoder(self, action: Any) -> Any:
        return bool(action) if action != 2 else None


EXPORT = HodgepodgeHelper
DESCRIPT = 'hodgepodge! everything!'
