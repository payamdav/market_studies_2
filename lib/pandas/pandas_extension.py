import pandas as pd
import numpy as np
from lib.rates.rates import rate_load
from lib.trader.trader import Trader
from lib.indicators.linearRegression import linearRegression, linearRegression_predict, linearRegression_predict_upto
from lib.utils.timer import TimerProfiler
from KDEpy import FFTKDE, NaiveKDE, TreeKDE
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from lib.indicators.labelBySlope import labelBySlope


def ewm_weights(period, *,alpha=None, com=None, span=None):
  if all([x is None for x in [alpha, com, span]]):
    span = period
  if alpha is None:
    if com is None:
      alpha = 2 / (span + 1)
    else:
      alpha = 1 / (1 + com)
  r = (1 - alpha)**np.arange(period)
  return r[::-1]


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
      't': self.df.t.to_numpy(),
      'o': self.df.o.to_numpy(),
      'h': self.df.h.to_numpy(),
      'l': self.df.l.to_numpy(),
      'c': self.df.c.to_numpy(),
      'p': 'c',
      'spread': self.df.s.to_numpy() / 100000,
    }
    t = Trader(**(defaults | kw))
    return t
  
  def forward_min_max_index(self, period):
    min_index, min_val = self.df.l.ext.forward_min_index(period)
    max_index, max_val = self.df.h.ext.forward_max_index(period)
    return min_index, max_index, min_val, max_val

  def expectancy(self, period, p_step=0.00001, bw=0.001):
    d = self.df[['h', 'l', 'c']].to_numpy().reshape(-1)
    dv = self.df[['v', 'v', 'v']].to_numpy().reshape(-1)
    idx = (np.arange(len(d))[:, None] - np.arange(3 * period-1, -1, -1)[None, :] + 2)[::3]
    idx = np.clip(idx, 0, len(d) - 1)
    samples = d[idx]
    weights = dv[idx]
    r = np.array([0.0] * len(samples))
    for i in range(len(samples)):
      p_min, p_max = samples[i].min(), samples[i].max()
      p_range = np.arange(p_min, p_max + p_step, p_step)
      kernel = NaiveKDE(kernel='gaussian', bw=bw).fit(samples[i], weights=weights[i])
      kde = kernel.evaluate(p_range)
      r[i] = (p_range @ kde) / kde.sum()
    return pd.Series(r, index=self.df.index, name=f"expectancy{period}")
    
  def expectancy_check(self, period, p_step=0.00001, bw=0.001):
    d = self.df[['h', 'l', 'c']].to_numpy().reshape(-1)
    dv = self.df[['v', 'v', 'v']].to_numpy().reshape(-1)
    idx = (np.arange(len(d))[:, None] - np.arange(3 * period-1, -1, -1)[None, :] + 2)[::3]
    idx = np.clip(idx, 0, len(d) - 1)
    samples = d[idx]
    weights = dv[idx]
    i = period + 10
    p_min, p_max = samples[i].min(), samples[i].max()
    p_range = np.arange(p_min, p_max + p_step, p_step)
    kernel = NaiveKDE(kernel='gaussian', bw=bw).fit(samples[i], weights=weights[i])
    kde = kernel.evaluate(p_range)
    r = (p_range @ kde) / kde.sum()
    print(f"p_min: {p_min}, p_max: {p_max}, r: {r}")
    fig, ax = plt.subplots(figsize=(18,8), nrows=1, ncols=1)
    # secax.plot(kde, p_range, 'r')
    plt.barh(p_range, kde, height=p_step/2, color='r', alpha=0.8)

    Cursor(ax, useblit=True, color='red', linewidth=2)

    plt.show()
  
  def kde_anal(self, p_step=0.0001, bw=0.001):
    timer = TimerProfiler('pandas_extension')
    timer.checkpoint('start')
    d = self.df[['h', 'l', 'c']].to_numpy().reshape(-1)
    dv = self.df[['v', 'v', 'v']].to_numpy().reshape(-1)
    p_min, p_max = d.min(), d.max()
    p_range = np.arange(p_min, p_max + p_step, p_step)
    timer.checkpoint('data prepare')
    kernel = TreeKDE(kernel='gaussian', bw='ISJ').fit(d, weights=dv)
    timer.checkpoint('kde fit')
    kde = kernel.evaluate(p_range)
    timer.checkpoint('kde evaluate')
    r = (p_range @ kde) / kde.sum()
    timer.checkpoint('r calc')
    print(f"p_min: {p_min}, p_max: {p_max}, r: {r}")
    print(f"Kernel bw: {kernel.bw}")
    # print(NaiveKDE._bw_methods.keys())
    fig, ax = plt.subplots(figsize=(18,8), nrows=1, ncols=1)
    p = (self.df.h + self.df.l + self.df.c) / 3
    ax.plot(self.df.index, p, 'b', markersize=2)
    ax.set_xlabel('Index')
    ax.set_ylabel('Price')
    secax = ax.twiny()
    secax.set_xlabel('KDE')
    secax.set_xmargin(2)
    # secax.barh(p_range, kde, height=p_step/2, color='r', alpha=0.8)
    # secax.barh(p_range, kde, height=p_step/5, color='r', alpha=0.8)
    secax.plot(kde, p_range, 'r')
    timer.checkpoint('plot')
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    plt.show()
  
  def atr(self, period):
    # average true range
    tr = self.df.h - self.df.l
    tr = pd.concat([tr, (self.df.h - self.df.c.shift(1)).abs(), (self.df.l - self.df.c.shift(1)).abs()], axis=1).max(axis=1)
    temp_atr = tr.ewm(span=period, adjust=False).mean()
    return pd.Series(temp_atr, index=self.df.index, name=f"atr{period}")
  
  def csize(self, period):
    # average candles size
    temp = self.df.h - self.df.l
    temp_ma = temp.ext.ma(period)
    temp_ma.name = f"csize{period}"
    return temp_ma
  
  def cbsize(self, period):
    # average candles body size
    temp = (self.df.o - self.df.c).abs()
    temp_ma = temp.ext.ma(period)
    temp_ma.name = f"cbsize{period}"
    return temp_ma
  
  


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
  
  def wma(self, period=1, weights=None):
    idx = np.arange(len(self.s))[:, None] - np.arange(period-1, -1, -1)[None, :]
    idx = np.clip(idx, 0, len(self.s) - 1)
    return pd.Series(np.average(self.s.to_numpy()[idx], 1, weights=weights.to_numpy()[idx]), index=self.s.index, name=f"{self.s.name}_wma{period}")

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
  
  def ema(self, period):
    return self.s.ewm(span=period, adjust=False).mean()
  
  def wema(self, period, weights):
    return self.s.ext.wma(period, weights=weights).ewm(span=period, adjust=False).mean()
  
  def labeler_by_slope(self, period):
    return pd.Series(labelBySlope(self.s, period)[0], index=self.s.index, name=f"{self.s.name}_labeler_by_slope{period}")

  def labeler_by_slope_degrees(self, period):
    return pd.Series(labelBySlope(self.s, period)[1], index=self.s.index, name=f"{self.s.name}_labeler_by_slope_degrees{period}")

  def labeler_by_slope_sines(self, period):
    return pd.Series(labelBySlope(self.s, period)[2], index=self.s.index, name=f"{self.s.name}_labeler_by_slope_sines{period}")
   