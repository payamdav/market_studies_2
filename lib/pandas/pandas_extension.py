import pandas as pd
import numpy as np
from lib.rates.rates import rate_load
from lib.trader.trader import Trader
from lib.indicators.linearRegression import linearRegression, linearRegression_predict, linearRegression_predict_upto


@pd.api.extensions.register_dataframe_accessor("ext")
class LibsAccessor():
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
  
  def forward_min_max_index(self, period):
    min_index, min_val = self.df.l.ext.forward_min_index(period)
    max_index, max_val = self.df.h.ext.forward_max_index(period)
    return min_index, max_index, min_val, max_val
  


# Pandas Series Extension

@pd.api.extensions.register_series_accessor("ext")
class LibsAccessorSerries():
  def __init__(self, s):
    self.s = s

  def previous(self, period=1):
    idx = np.arange(len(self.s))[:, None] - np.arange(period, 0, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    temp = self.s.to_numpy()[idx]
    temp[0:period] = np.nan
    return pd.DataFrame(temp, index=self.s.index, columns=[f"{self.s.name}_B{i}" for i in range(1, period + 1)])
  
  def apply_on_previous(self, period, func, **kwargs):
    idx = np.arange(len(self.s))[:, None] - np.arange(period-1, -1, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    temp = self.s.to_numpy()[idx]
    r = np.apply_along_axis(func, 1, temp, **kwargs)
    r[0:period-1] = np.nan
    return pd.Series(r, index=self.s.index, name=f"{self.s.name}_apply_on_previous{period}")

  def ma(self, period=1):
    idx = np.arange(len(self.s))[:, None] - np.arange(period-1, -1, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    temp = self.s.to_numpy()[idx]
    return pd.Series(temp.mean(axis=1), index=self.s.index, name=f"{self.s.name}_ma{period}")
  
  def slope(self, period=1, x_step=0.0001):
    idx = np.arange(len(self.s))[:, None] - np.arange(period-1, -1, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    y = self.s.to_numpy()[idx]
    Y = y - y.mean(axis=1)[:, None]
    x = (np.arange(period) * x_step)
    X = x - x.mean()
    slopes = np.dot(Y, X) / np.dot(X, X )
    return pd.Series(slopes, index=self.s.index, name=f"{self.s.name}_slope{period}")
  
  def sine_slope(self, period=1, x_step=0.0001):
    idx = np.arange(len(self.s))[:, None] - np.arange(period-1, -1, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    y = self.s.to_numpy()[idx]
    Y = y - y.mean(axis=1)[:, None]
    x = (np.arange(period) * x_step)
    X = x - x.mean()
    slopes = np.dot(Y, X) / np.dot(X, X )
    sines = slopes / np.sqrt(1 + np.power(slopes, 2))
    return pd.Series(sines, index=self.s.index, name=f"{self.s.name}_sine_slope{period}")
    
  def lsma(self, period=1):
    idx = np.arange(len(self.s))[:, None] - np.arange(period-1, -1, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    y = self.s.to_numpy()[idx]
    temp = linearRegression_predict(y, 0)
    return pd.Series(temp, index=self.s.index, name=f"{self.s.name}_lsma{period}")
  
  def lr(self, period=1, xStep=1):
    idx = np.arange(len(self.s))[:, None] - np.arange(period-1, -1, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    y = self.s.to_numpy()[idx]
    slopes, intercepts = linearRegression(y, xStep)
    return pd.DataFrame({f"{self.s.name}_slope{period}": slopes, f"{self.s.name}_intercept{period}": intercepts}, index=self.s.index)
  
  def lr_predict(self, period=1, next=1):
    idx = np.arange(len(self.s))[:, None] - np.arange(period-1, -1, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    y = self.s.to_numpy()[idx]
    temp = linearRegression_predict(y, next)
    return pd.Series(temp, index=self.s.index, name=f"{self.s.name}_lr_predict{period}_{next}")
  
  def lr_predict_upto(self, period=1, upto=1):
    idx = np.arange(len(self.s))[:, None] - np.arange(period-1, -1, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    y = self.s.to_numpy()[idx]
    temp = linearRegression_predict_upto(y, upto)
    return pd.DataFrame(temp, index=self.s.index, columns=[f"{self.s.name}_lr_predict{period}_{i}" for i in range(1, upto + 1)])
  
  def forward_min_index(self, period):
    idx = np.arange(len(self.s))[:, None] + np.arange(0, period)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    temp = self.s.to_numpy()[idx]
    min_index = np.argmin(temp, axis=1) + self.s.index
    min_val = self.s.to_numpy()[min_index]
    return pd.Series(min_index, index=self.s.index, name="min_index"), pd.Series(min_val, index=self.s.index, name="min_val")
  
  def forward_max_index(self, period):
    idx = np.arange(len(self.s))[:, None] + np.arange(0, period)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    temp = self.s.to_numpy()[idx]
    max_index = np.argmax(temp, axis=1) + self.s.index
    max_val = self.s.to_numpy()[max_index]
    return pd.Series(max_index, index=self.s.index, name="max_index"), pd.Series(max_val, index=self.s.index, name="max_val")

  def forward_min_max_index(self, period):
    idx = np.arange(len(self.s))[:, None] + np.arange(0, period)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    temp = self.s.to_numpy()[idx]
    min_index = np.argmin(temp, axis=1) + self.s.index
    max_index = np.argmax(temp, axis=1) + self.s.index
    min_val = self.s.to_numpy()[min_index]
    max_val = self.s.to_numpy()[max_index]
    return pd.Series(min_index, index=self.s.index, name="min_index"), pd.Series(max_index, index=self.s.index, name="max_index"), pd.Series(min_val, index=self.s.index, name="min_val"), pd.Series(max_val, index=self.s.index, name="max_val")
