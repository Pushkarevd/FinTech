import pandas as pd
import warnings
import yfinance as yahooFinance

warnings.filterwarnings("ignore")


def calculate_rsi(data: pd.Series, window: int) -> pd.Series:
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window).mean()
    average_loss = abs(down.rolling(window).mean())
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(data: pd.Series, window: int) -> tuple:
    rolling_mean = data.rolling(window).mean()
    rolling_std = data.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band


def calculate_obv(data: pd.DataFrame) -> list:
    obv = [0]
    for i in range(1, len(data.Close)):
        if data.Close[i] > data.Close[i - 1]:
            obv.append(obv[-1] + data.Volume[i])
        elif data.Close[i] < data.Close[i - 1]:
            obv.append(obv[-1] - data.Volume[i])
        else:
            obv.append(obv[-1])
    return obv


def strategy(df: pd.DataFrame) -> tuple:
    buy, sell = [], []
    last_signal = None
    for index, row in df.iterrows():
        if last_signal is None or last_signal == 'SELL':
            if row['RSI'] > 50 and row['Close'] > row['Rolling Mean'] and row['OBV'] > 0:
                buy.append(index)
                last_signal = 'BUY'
        else:
            if row['Close'] < row['Bollinger Low']:
                sell.append(index)
                last_signal = 'SELL'
    return buy, sell


def prepare_data(df: pd.DataFrame, window=14) -> pd.DataFrame:
    df['RSI'] = calculate_rsi(df['Close'], window)
    df['Rolling Mean'], df['Bollinger High'], df['Bollinger Low'] = calculate_bollinger_bands(df['Close'], window)
    df['OBV'] = calculate_obv(df)
    return df


if __name__ == "__main__":
    ticker = input('Введите наименование тикера: ')
    try:
        df = yahooFinance.Ticker(ticker).history(period='5y')
    except Exception as err:
        print('Тикер неверный или не найден на платформе Yahoo Finance')
        raise Exception(err)

    df = prepare_data(df)
    buy, sell = strategy(df)

    buy_price = [df.loc[x]['Close'] for x in buy]
    sell_price = [df.loc[x]['Close'] for x in sell]

    profit = 0
    for idx in range(len(sell_price)):
        profit += sell_price[idx] - buy_price[idx]

    result = pd.DataFrame(zip(buy_price, sell_price), columns=['Цена покупки', 'Цена продажи'])
    result['Прибыль'] = result['Цена продажи'] - result['Цена покупки']
    print(result)

    print(f'Полученный профит {profit:.2f}')
    
