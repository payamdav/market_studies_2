import pandas as pd
import numpy as np
from lib.rates.rates import rate_load
from lib.trader.trader import Trader


@pd.api.extensions.register_dataframe_accessor("libs")
class LibsAccessor:
  def __init__(self, df):
    self.df = df
  
  def load(self, pair):
    return rate_load(pair)

  def resample_candles(self, freq):
    temp = self.df.resample(freq, on='t').agg({'ts': 'min', 'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum', 's': 'last', 'r': 'sum'})
    temp['t'] = temp.index
    temp.reset_index(drop=True, inplace=True)
    temp.sort_values(by='t', inplace=True)
    temp.dropna(inplace=True)
    temp.reset_index(drop=True, inplace=True)
    return temp

  def insert_previous(self, column_name, previous=1):
    idx = np.arange(len(self.df))[:, None] - np.arange(1, previous + 1)[None, :]
    idx = np.clip(idx, 0, len(self.df) - 1)
    temp = self.df[column_name].to_numpy()[idx]
    temp[0:previous] = np.nan
    return pd.concat([self.df, pd.DataFrame(temp, columns=[f"{column_name}_B{i}" for i in range(1, previous + 1)])], axis=1)
  
  def apply_on_previous(self, column_name, previous, func, **kwargs):
    idx = np.arange(len(self.df))[:, None] - np.arange(0, previous)[None, :]
    idx = np.clip(idx, 0, len(self.df) - 1)
    temp = self.df[column_name].to_numpy()[idx]
    r = np.apply_along_axis(func, 1, temp, **kwargs)
    r[0:previous-1] = np.nan
    return r

  def insert_slope(self, column_name, previous=1, x_range=0.0001):
    idx = np.arange(len(self.df))[:, None] - np.arange(0, previous)[None, :]
    idx = np.clip(idx, 0, len(self.df) - 1)
    y = self.df[column_name].to_numpy()[idx]
    Y = y - y.mean(axis=1)[:, None]
    x = np.linspace(0, x_range * previous, previous)[::-1]
    X = x - x.mean()
    slopes = np.dot(Y, X) / np.dot(X, X )
    return pd.concat([self.df, pd.DataFrame(slopes, columns=[f"{column_name}_slope_{previous}"])], axis=1)

  def insert_sine_slope(self, column_name, previous=1, x_range=0.0001):
    idx = np.arange(len(self.df))[:, None] - np.arange(0, previous)[None, :]
    idx = np.clip(idx, 0, len(self.df) - 1)
    y = self.df[column_name].to_numpy()[idx]
    Y = y - y.mean(axis=1)[:, None]
    x = np.linspace(0, x_range * previous, previous)[::-1]
    X = x - x.mean()
    slopes = np.dot(Y, X) / np.dot(X, X )
    sines = slopes / np.sqrt(1 + np.power(slopes, 2))
    return pd.concat([self.df, pd.DataFrame(sines, columns=[f"{column_name}_sine_{previous}"])], axis=1)
  
  def insert_eucledian_distance(self, column_names):
    if isinstance(column_names, str):
      column_names = [column_names]
    if column_names is None:
      column_names = self.df.columns
    temp = np.linalg.norm(self.df[column_names], axis=1)
    tempdf = pd.DataFrame(temp)
    tempdf.columns = [f"eucledian_{'_'.join(column_names)}"]
    return pd.concat([self.df, tempdf], axis=1)
  
  def insert_normalize_rows(self, column_names):
    if isinstance(column_names, str):
      column_names = [column_names]
    if column_names is None:
      column_names = self.df.columns
    norm = np.linalg.norm(self.df[column_names], axis=1)
    temp = pd.DataFrame(self.df[column_names]/norm[:, None])
    temp.columns = [f"{column_name}_norm" for column_name in column_names]
    return pd.concat([self.df, temp], axis=1)
  
  def trade(self, **kw):
    defaults = {
      'o': self.df.o.to_numpy(),
      'h': self.df.h.to_numpy(),
      'l': self.df.l.to_numpy(),
      'c': self.df.c.to_numpy(),
      'p': 'c',
    }
    t = Trader(**(defaults | kw))
    return t
