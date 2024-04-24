from ._basic_data_locator import BasicDataLocator
import pandas as pd
import talib
from FinMind.data import DataLoader


def apply_ii(y):
    if float(y) > 0:
        return 1
    elif float(y) < 0:
        return -1
    else:
        return 0


class HodgepodgeLocator(BasicDataLocator):
    def __init__(self, *args, **kwargs):
        super(HodgepodgeLocator, self).__init__(*args, **kwargs)

    def _get_current_institutional_investor_data(self) -> pd.DataFrame:
        if self.online:
            dl = DataLoader()
            if self.config['FINMIND_TOKEN'] is not None:
                dl.login_by_token(self.config['FINMIND_TOKEN'])
            data = dl.taiwan_stock_institutional_investors(stock_id=self.stock_id[0],
                                                           start_date=self.start,
                                                           end_date=self.end)
            data = data.set_index('date')
        else:
            data = pd.read_csv(f'{self.data_root}/法人買賣超/{self.index[self.offset]}.csv')
            data = data[data['date'] >= str(self.start)]
            data = data[data['date'] <= str(self.end)]
            data = data.reset_index(drop=True).set_index('date')

        data = data.groupby(['date', 'name']).agg({'buy': sum, 'sell': sum})
        data = data.apply(lambda x: x['buy'] - x['sell'], axis=1)
        data = data.unstack('name')
        if 'Foreign_Dealer_Self' not in data.columns:
            data['Foreign_Dealer_Self'] = 0
        data = data[['Foreign_Investor', 'Foreign_Dealer_Self', 'Investment_Trust', 'Dealer_self', 'Dealer_Hedging']]
        data['Dealer_total'] = data.apply(lambda x: x['Dealer_self'] + x['Dealer_Hedging'], axis=1)
        data['institutional_investors'] = data.apply(lambda x: sum(x), axis=1)
        data = data.apply(lambda x: x.apply(apply_ii))

        data = data.shift(1)  # 往下偏移一天 (因為當天結束才會統計買賣超資訊，實務上交易日當天是不知道當天買賣超資訊的)
        data = data.fillna(0)  # 用 0 補空值 (影響應該不會太大)

        return data

    def next(self) -> pd.DataFrame:
        data = self._get_current_data()

        ii = self._get_current_institutional_investor_data()

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        data = pd.concat([data, ii], axis=1).dropna()

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
                     'Foreign_Investor', 'Foreign_Dealer_Self', 'Investment_Trust',
                     'Dealer_total', 'Dealer_self', 'Dealer_Hedging', 'institutional_investors',  # 法人買賣超
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
DESCRIPT = 'Hodgepodge Locator v3'

if __name__ == '__main__':
    raise NotImplementedError('this file not meant to be executed')
