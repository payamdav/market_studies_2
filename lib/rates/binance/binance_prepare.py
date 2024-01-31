import os
import pathlib
import urllib.request
import shutil
import zipfile
import pandas as pd
import numpy as np

raw_files_path = os.path.join(os.path.dirname(__file__), "..", "..", "files", "rates", "raw", "binance")
rates_path = os.path.join(os.path.dirname(__file__), "..", "..", "files", "rates")

# config
class ConfigClass:
    pass
config = ConfigClass()

config.binance_url_spot = "https://data.binance.vision/data/spot/monthly/klines"
config.binance_url_futures = "https://data.binance.vision/data/futures/um/monthly/klines"
config.path = os.path.join(os.path.dirname(__file__), "..", "..", "files", "rates", "raw", "binance_archive")
config.futures_path = os.path.join(os.path.dirname(__file__), "..", "..", "files", "rates", "raw", "binance_archive", "futures", "candles")
config.spot_path = os.path.join(os.path.dirname(__file__), "..", "..", "files", "rates", "raw", "binance_archive", "spot", "candles")

if not os.path.exists(path=config.path):
    os.makedirs(config.path)
if not os.path.exists(path=config.futures_path):
    os.makedirs(config.futures_path)
if not os.path.exists(path=config.spot_path):
    os.makedirs(config.spot_path)



def year_month_iterator(year, month, to_year=0, to_month=0):
    to_year = year if to_year == 0 else to_year
    to_month = month if to_month == 0 else to_month
    for ym in range(12 * year + month - 1, 12 * to_year + to_month - 1 + 1):
        y, m = divmod(ym, 12)
        m += 1
        yield y, m


def cumulative_file_path(market, symbol):
    symbol = str(symbol).strip().upper()
    market = str(market).strip().lower()
    base_path = config.futures_path if market == 'futures' else config.spot_path
    return os.path.join(base_path, symbol + '_1min')


def download_binance_historical_candles_single(market, symbol, timeframe, year, month):
    try:
        symbol = str(symbol).strip().upper()
        market = str(market).strip().lower()
        timeframe = str(timeframe).strip().lower()
        year = str(year).strip()
        month = str(month).strip().zfill(2)
        binance_base_url = config.binance_url_futures if market == 'futures' else config.binance_url_spot
        url = f"{binance_base_url}/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month}.zip"
        file_name = f"{symbol}-{timeframe}-{year}-{month}.zip"
        base_path = config.futures_path if market == 'futures' else config.spot_path
        file_path = os.path.join(base_path, file_name)
        with urllib.request.urlopen(url) as response, open(file_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(base_path)
        os.remove(file_path)
    except Exception as e:
        print("Error", e)


def download_binance_historical_candles(market, symbol, timeframe, year, month, to_year=0, to_month=0):
    try:
        for y, m in year_month_iterator(year, month, to_year, to_month):
            download_binance_historical_candles_single(market, symbol, timeframe, y, m)
    except Exception as e:
        print("Error", e)


def get_pandas_df_binance_historical_candles_from_local(market, symbol, timeframe, year, month, to_year=0, to_month=0):
    try:
        binance_to_pandas_freq = {
          '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', '30m': '30T', '1h': 'H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H', '1d': 'D', '3d': '3D', '1w': 'W', '1mo': 'M'
        }
        header_names = ['ts', 'o', 'h', 'l', 'c', 'v', 'Close_Time', 'q', 'n', 'TVolume', 'Tq', 'ignore']
        use_columns = ['ts', 'o', 'h', 'l', 'c', 'v', 'n']
        symbol = str(symbol).strip().upper()
        market = str(market).strip().lower()
        timeframe = str(timeframe).strip().lower()
        all_candles = pd.DataFrame()
        for y, m in year_month_iterator(year, month, to_year, to_month):
          file_name = f"{symbol}-{timeframe}-{str(y).strip()}-{str(m).strip().zfill(2)}.csv"
          base_path = config.futures_path if market == 'futures' else config.spot_path
          file_path = os.path.join(base_path, file_name)
          candles = pd.read_csv(file_path, names=header_names, usecols=use_columns, dtype={'ts': 'int64', 'o': 'float64', 'h': 'float64', 'l': 'float64', 'c': 'float64', 'v': 'float64', 's': 'float64', 'n': 'int64'}, header=0)
          candles["t"] = pd.to_datetime(candles["ts"], unit="ms")
          all_candles = pd.concat([all_candles, candles])
          # all_candles = candles.combine_first(all_candles)
        all_candles.reset_index(drop=True, inplace=True)
        all_candles['s'] = all_candles.c * 100
        all_candles.to_feather(os.path.join(config.path, f"{symbol}.feather"))
        

    except Exception as e:
        print("Error", e)



# download_binance_historical_candles(market='futures', symbol='BTCUSDT', timeframe='1m', year=2023, month=12, to_year=2023, to_month=12)
# download_binance_historical_candles(market='futures', symbol='ETHUSDT', timeframe='1m', year=2022, month=2, to_year=2022, to_month=2)
# get_pandas_df_binance_historical_candles_from_local(market='futures', symbol='ETHUSDT', timeframe='1m', year=2022, month=1, to_year=2023, to_month=12)
df = pd.read_feather(os.path.join(config.path, f"BTCUSDT.feather"))
df['r'] = df.v
df.to_feather(os.path.join(config.path, f"BTCUSDT.feather"))
print(df)
print(df.info())
d = df.t - df.t.shift(1)
neg = d < pd.Timedelta(0)
otherthan1min = d != pd.Timedelta(minutes=1)
print(f"Found {neg.sum()} negative time deltas")
print(f"Found {otherthan1min.sum()} other than 1 minute time deltas")
print('index of other than 1 minute time deltas: ', df[otherthan1min].index)
print("rows with nan: ", df.shape[0] - df.dropna().shape[0])

