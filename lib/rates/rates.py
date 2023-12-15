import pandas as pd
import os
import pathlib

raw_files_path = os.path.join(os.path.dirname(__file__), "..", "..", "files", "rates", "raw", "ecn_mt5")
rates_path = os.path.join(os.path.dirname(__file__), "..", "..", "files", "rates")
rates_base_url = "https://github.com/payamdav/rates/blob/main/"

def build_rates_feather(pair, suffix):
  print(f"Building rates feather file from {pair}")
  df = pd.read_csv(os.path.join(raw_files_path, f"{pair}{suffix}.csv"), names=['t', 'ts', 'o', 'h', 'l', 'c', 'v', 's', 'r'], 
                   header=None,
                   dtype={'t': 'str', 'ts': 'int64', 'o': 'float64', 'h': 'float64', 'l': 'float64', 'c': 'float64', 'v': 'float64', 's': 'float64', 'r': 'float64'},
                   )
  df['t'] = pd.to_datetime(df['t'])
  d = df.t - df.t.shift(1)
  neg = d < pd.Timedelta(0)
  print(f"Found {neg.sum()} negative time deltas")
  print("rows with nan: ", df.shape[0] - df.dropna().shape[0])
  df.to_feather(os.path.join(rates_path, f"{pair}.feather"))


def rate_load_local(pair):
  pair = pair.upper()
  return pd.read_feather(os.path.join(rates_path, f"{pair}.feather"))

def rate_load(pair):
  pathlib.Path(rates_path).mkdir(parents=True, exist_ok=True)
  pair = pair.upper()
  url = f"{rates_base_url}{pair}.feather?raw=true"
  local_path = os.path.join(rates_path, f"{pair}.feather")
  if os.path.isfile(local_path):
    print(f"Found local {pair}.feather file")
    return rate_load_local(pair)
  else:
    print(f"Downloading {pair}.feather file from {url}")
    df = pd.read_feather(url)
    df.to_feather(local_path)
    return df



# build_rates_feather("AUDCAD", "_data")
# build_rates_feather("AUDUSD", "_data")
# build_rates_feather("EURAUD", "_data")
# build_rates_feather("EURCAD", "_data")
# build_rates_feather("EURUSD", "_data")
# build_rates_feather("NZDCAD", "_data")
# build_rates_feather("USDCAD", "_data")

# df = rate_load("EURUSD")
# print(df)
# print(df.info())



