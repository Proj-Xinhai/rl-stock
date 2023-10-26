from ._basic_data_locator import BasicDataLocator
import pandas as pd
import talib


class HodgepodgeLocator(BasicDataLocator):
    def __init__(self, *args, **kwargs):
        super(HodgepodgeLocator, self).__init__(*args, **kwargs)

    def next(self) -> pd.DataFrame:
        data = self._get_current_data()

        ii = pd.read_csv(f'{self.data_root}/法人買賣超日報_個股/{self.index[self.offset]}.csv', index_col=0)
        ii = ii.reset_index(drop=True).set_index('Date')

        ii = ii.apply(lambda x: x.apply(lambda y: y.replace(',', '') if type(y) == str else y))
        ii = ii[['外陸資買賣超股數(不含外資自營商)', '外資自營商買賣超股數', '投信買賣超股數', '自營商買賣超股數',
                 '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)', '三大法人買賣超股數']]
        ii = ii.apply(lambda x: x.apply(lambda y: 1 if float(y) > 0 else -1 if float(y) < 0 else 0))
        ii = ii.shift(1)  # 往下偏移一天 (因為當天結束才會統計買賣超資訊，實務上交易日當天是不知道當天買賣超資訊的)
        ii = ii.fillna(0)  # 用 0 補空值 (影響應該不會太大)

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        data = pd.concat([data, ii], axis=1).dropna()

        # 合併完，再計算技術指標 (避免 dropna 把技術指標開頭資料刪除)
        # SMA
        data['MA5'] = talib.MA(data['Close'], timeperiod=5)
        data['MA10'] = talib.MA(data['Close'], timeperiod=10)
        data['MA20'] = talib.MA(data['Close'], timeperiod=20)
        data['MA60'] = talib.MA(data['Close'], timeperiod=60)
        # MACD
        data['MACD'], data['MACDsignal'], data['MACDhist'] = \
            talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
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

        data = data.fillna(0)  # 補空值

        if self.offset < len(self.index) - 1:
            self.offset += 1
        else:
            self._set_index(random_state=self.random_state)
            self.offset = 0

        return data


EXPORT = HodgepodgeLocator
DESCRIPT = 'Hodgepodge Locator'

if __name__ == '__main__':
    raise NotImplementedError('this file not meant to be executed')
