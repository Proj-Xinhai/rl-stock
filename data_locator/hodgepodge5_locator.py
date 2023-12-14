from ._basic_data_locator import BasicDataLocator
import pandas as pd
import talib
from FinMind.data import DataLoader


def divide(short_sale_today_balance, margin_purchase_today_balance):
    if margin_purchase_today_balance == 0:
        return 0
    return short_sale_today_balance / margin_purchase_today_balance


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

        data = data.groupby(['date']).agg({'buy': sum, 'sell': sum})
        data = data.apply(lambda x: x['buy'] - x['sell'], axis=1).to_frame()
        data.columns = ['buy_sell']
        data = data.apply(lambda x: x.apply(lambda y: 1 if float(y) > 0 else -1 if float(y) < 0 else 0))

        # data = data.shift(1)  # 往下偏移一天 (因為當天結束才會統計買賣超資訊，實務上交易日當天是不知道當天買賣超資訊的)
        # data = data.fillna(0)  # 用 0 補空值 (影響應該不會太大)

        return data

    def _get_margin_purchase_short_sale_ratio_data(self) -> pd.DataFrame:
        if self.online:
            dl = DataLoader()
            if self.config['FINMIND_TOKEN'] is not None:
                dl.login_by_token(self.config['FINMIND_TOKEN'])
            data = dl.taiwan_stock_margin_purchase_short_sale(stock_id=self.stock_id[0],
                                                              start_date=self.start,
                                                              end_date=self.end)
            data = data.set_index('date')
        else:
            data = pd.read_csv(f'{self.data_root}/融資融券/{self.index[self.offset]}.csv')
            data = data[data['date'] >= str(self.start)]
            data = data[data['date'] <= str(self.end)]
            data = data.reset_index(drop=True).set_index('date')

        data = data.apply(lambda x: divide(x['ShortSaleTodayBalance'], x['MarginPurchaseTodayBalance']), axis=1).to_frame()
        data.columns = ['short_sale_margin_purchase_ratio']

        # data = data.shift(1)  # 往下偏移一天 (因為當天結束才會統計買賣超資訊，實務上交易日當天是不知道當天買賣超資訊的)
        # data = data.fillna(0)  # 用 0 補空值 (影響應該不會太大)

        return data

    def next(self) -> pd.DataFrame:
        data = self._get_current_data()

        ii = self._get_current_institutional_investor_data()
        margin_short = self._get_margin_purchase_short_sale_ratio_data()

        # 合併資料，並把含有空值之列刪除
        # 空值原因: 部分補班日不開盤，但是含有法人買賣超資料，此時個股資料會有空值
        data = pd.concat([data, ii, margin_short], axis=1).dropna()

        # 當日若無資料，該列不存在，因此補 0
        data['buy_sell'] = data['buy_sell'].fillna(0)
        data['short_sale_margin_purchase_ratio'] = data['short_sale_margin_purchase_ratio'].fillna(0)

        # 下移一天
        data['buy_sell'] = data['buy_sell'].shift(1)
        data['short_sale_margin_purchase_ratio'] = data['short_sale_margin_purchase_ratio'].shift(1)

        # 補 0
        data['buy_sell'] = data['buy_sell'].fillna(0)
        data['short_sale_margin_purchase_ratio'] = data['short_sale_margin_purchase_ratio'].fillna(0)

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
                     'buy_sell',  # 法人買賣超
                     'short_sale_margin_purchase_ratio',  # 券資比
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
DESCRIPT = 'Hodgepodge Locator v5'

if __name__ == '__main__':
    raise NotImplementedError('this file not meant to be executed')
