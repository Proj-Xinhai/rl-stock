from ._basic_data_locator import BasicDataLocator
import pandas as pd
import talib


class HodgepodgeLocator(BasicDataLocator):
    def __init__(self, *args, **kwargs):
        super(HodgepodgeLocator, self).__init__(*args, **kwargs)

    def next(self) -> pd.DataFrame:
        data = self._get_current_data()

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        data = pd.concat([data], axis=1).dropna()

        # 合併完，再計算技術指標 (避免 dropna 把技術指標開頭資料刪除)
        # MACD
        data['MACD'], data['MACDsignal'], data['MACDhist'] = \
            talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        # RSI
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)
        # CCI
        data['CCI'] = talib.CCI(data['max'], data['min'], data['close'], timeperiod=14)
        # ADX
        data['ADX'] = talib.ADX(data['max'], data['min'], data['close'], timeperiod=14)

        data = data[['close',  # 收盤價
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
DESCRIPT = 'Hodgepodge Locator v2'

if __name__ == '__main__':
    raise NotImplementedError('this file not meant to be executed')
