import numpy as np
import sys
from types import SimpleNamespace
from lib.trader.hypertrader import hyperTrader
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from plotly import graph_objects as go
from plotly.offline import iplot

property_type = SimpleNamespace(
  absolute=0,
  percent=1,
  delta=2,
)

exit_reasons = SimpleNamespace(
  noTrade=-1,
  unknown=0,
  sl=1,
  tp=2,
  limit=3,
)

class Trader:
  def __init__(self, **kw):
    defaults = {
      't': None,
      'o': None,
      'h': None,
      'l': None,
      'c': None,
      'p': None,
      'spread': None,

      'd': None,
      'tp': None,
      'sl': None,
      'limit': None,
      'trail': None,
      'trail_activation': None,
      'risk_free': None,

      'sl_type': property_type.absolute,
      'tp_type': property_type.absolute,
    }

    self.__dict__.update(defaults | kw)
    self.n = 0
  
  def init(self):
    if self.c is None and self.p is None:
      raise Exception('p or c must be provided')
    if self.c is None:
      self.o = self.h = self.l = self.c = self.p
    if self.p is None:
      self.p = self.c.copy()
    if self.p in ['o', 'h', 'l', 'c']:
      self.p = self.__dict__[self.p]
    self.n = len(self.p)

    if self.d is None:
      self.d = np.full(self.n, 0, dtype=np.int32)
    elif isinstance(self.d, (int, float)):
      self.d = np.full(self.n, self.d, dtype=np.int32)
    elif isinstance(self.d, str):
      self.d = np.full(self.n, 1 if self.d == 'long' else -1, dtype=np.int32)
    if self.d.dtype != np.int32:
      self.d = self.d.astype(np.int32)

    if self.sl is None:
      self.sl = np.full(self.n, -1 * self.d * sys.float_info.max, dtype=float)
      self.sl_type = property_type.absolute
    elif isinstance(self.sl, (int, float)):
      self.sl = np.full(self.n, self.sl, dtype=float)
    if self.tp is None:
      self.tp = np.full(self.n, self.d * sys.float_info.max, dtype=float)
      self.tp_type = property_type.absolute
    elif isinstance(self.tp, (int, float)):
      self.tp = np.full(self.n, self.tp, dtype=float)
    if self.limit is None:
      self.limit = np.full(self.n, self.n, dtype=np.int32)
    elif isinstance(self.limit, (int, float)):
      self.limit = np.full(self.n, self.limit, dtype=np.int32)
    if self.limit.dtype != np.int32:
      self.limit = self.limit.astype(np.int32)
    if self.spread is None:
      self.spread = np.full(self.n, 0, dtype=float)
    elif isinstance(self.spread, (int, float)):
      self.spread = np.full(self.n, self.spread, dtype=float)
    if self.trail is None:
      self.trail = np.full(self.n, -1, dtype=float)
    elif isinstance(self.trail, (int, float)):
      self.trail = np.full(self.n, self.trail, dtype=float)
    if self.trail_activation is None:
      self.trail_activation = np.full(self.n, 0, dtype=float)
    elif isinstance(self.trail_activation, (int, float)):
      self.trail_activation = np.full(self.n, self.trail_activation, dtype=float)
    if self.risk_free is None:
      self.risk_free = np.full(self.n, -1, dtype=float)
    elif isinstance(self.risk_free, (int, float)):
      self.risk_free = np.full(self.n, self.risk_free, dtype=float)

    if self.sl_type == property_type.percent:
      self.sl = self.p - (self.d * (self.p * self.sl))
    elif self.sl_type == property_type.delta:
      self.sl = self.p - (self.d * self.sl)
    if self.tp_type == property_type.percent:
      self.tp = self.p + (self.d * (self.p * self.tp))
    elif self.tp_type == property_type.delta:
      self.tp = self.p + (self.d * self.tp)

    # check that all shapes must be same
    if not all([len(x) == self.n for x in [self.o, self.h, self.l, self.c, self.p, self.d, self.sl, self.tp, self.limit]]):
      raise Exception('d, sl, tp, limit must have same length')

    return self

  def doTrade(self):
    trade_indices = self.d.nonzero()[0]
    trade_count = len(trade_indices)
    entry = np.array(trade_indices, dtype=np.int32, copy=True)
    d = self.d[entry].copy()
    sl = self.sl[entry].copy()
    tp = self.tp[entry].copy()
    limit = self.limit[entry].copy()
    trail = self.trail[entry].copy()
    trail_activation = self.trail_activation[entry].copy()
    risk_free = self.risk_free[entry].copy()

    reason = np.full(trade_count, exit_reasons.unknown, dtype=np.int32)
    ex = np.full(trade_count, -1, dtype=np.int32)
    eprice = np.full(trade_count, 0, dtype=float)
    xprice = np.full(trade_count, 0, dtype=float)
    val = np.full(self.n, 0, dtype=float)
    eprofit = np.full(self.n, 0, dtype=float)
    num = np.full(self.n, 0, dtype=np.int32)
    win = np.full(self.n, 0, dtype=np.int32)

    hyperTrader(self.o.copy().data, self.h.copy().data, self.l.copy().data, self.c.copy().data, self.p.copy().data, d.data, entry.data, ex.data, reason.data, limit.data, tp.data, sl.data, eprice.data, xprice.data, val.data, num.data, self.spread.copy().data, eprofit.data, win.data, trail.data, trail_activation.data, risk_free.data)
    self.reason = np.full(self.n, exit_reasons.noTrade, dtype=np.int32)
    self.ex = np.full(self.n, -1, dtype=np.int32)
    self.eprice = np.full(self.n, 0, dtype=float)
    self.xprice = np.full(self.n, 0, dtype=float)
    self.entry = np.full(self.n, -1, dtype=np.int32)
    self.reason[entry] = reason
    self.ex[entry] = ex
    self.eprice[entry] = eprice
    self.xprice[entry] = xprice
    self.entry[entry] = entry
    self.val = val
    self.num = num
    self.win = win
    self.eprofit = eprofit

    return self
  
  def make_report(self):
    r = SimpleNamespace()
    r.count = self.d.nonzero()[0].size
    r.count_long = np.count_nonzero(self.d == 1)
    r.count_short = np.count_nonzero(self.d == -1)
    r.count_win = np.count_nonzero(self.win)
    r.count_loss = r.count - r.count_win
    r.count_limit = np.count_nonzero(self.reason == exit_reasons.limit)
    r.win_rate = r.count_win / r.count if r.count > 0 else 0
    r.loss_rate = r.count_loss / r.count if r.count > 0 else 0
    r.profit_long_no_spread = np.sum(self.xprice[self.d == 1] - self.eprice[self.d == 1])
    r.profit_short_no_spread = np.sum(self.eprice[self.d == -1] - self.xprice[self.d == -1])
    r.profit_no_spread = r.profit_long_no_spread + r.profit_short_no_spread
    r.profit_long = np.sum(self.eprofit[self.d == 1])
    r.profit_short = np.sum(self.eprofit[self.d == -1])
    r.profit = r.profit_long + r.profit_short
    r.total_spread = r.profit_no_spread - r.profit
    r.min_duration = np.min(self.ex[self.d != 0] - self.entry[self.d != 0])
    r.max_duration = np.max(self.ex[self.d != 0] - self.entry[self.d != 0])
    r.avg_duration = np.mean(self.ex[self.d != 0] - self.entry[self.d != 0])
    r.max_trade_count = np.max(self.num)
    r.max_portfolio_value = np.max(self.val)
    r.min_portfolio_value = np.min(self.val[self.val != 0])
    r.profit_win = np.sum(self.eprofit[self.win == 1])
    r.profit_loss = np.sum(self.eprofit[self.win == 0])
    r.average_profit_win = r.profit_win / r.count_win if r.count_win > 0 else 0
    r.average_profit_loss = r.profit_loss / r.count_loss if r.count_loss > 0 else 0
    r.average_profit = r.profit / r.count if r.count > 0 else 0
    self.report = r
    return self
  
  def print_report(self):
    for k, v in self.report.__dict__.items():
      print(f'\033[33m {k}: \033[97m {v}\033[00m ')
    return self
  
  def period_report(self, period):
    df = pd.DataFrame({'t': self.t, 'eprofit': self.eprofit, 'd': np.abs(self.d), 'win': self.win})
    r = df.groupby(pd.Grouper(key='t', freq=period)).sum().copy()
    r['win_rate'] = np.where(r['d'] > 0, r['win'] / r['d'], 0)
    r['profit'] = np.where(r['d'] > 0, r['eprofit'], 0)
    r['count'] = r['d']
    r = r[['count', 'win_rate', 'profit']]
    return r

  def plot(self):
    self.plot_orders()
    return self
  
  def plot_orders(self, shrink=True):
    i_start = 0
    i_end = self.n
    if shrink:
      i_start = np.min(self.entry[self.entry >= 0])
      i_end = np.max(self.ex[self.ex >= 0])
    layout = dict(title='Trade', xaxis_title='Time', yaxis_title='Price', yaxis=dict(autorange=True, fixedrange=False))
    candlesticks = dict(type='candlestick', name='Candlesticks', x=np.arange(i_start,i_end), open=self.o[i_start:i_end], high=self.h[i_start:i_end], low=self.l[i_start:i_end], close=self.c[i_start:i_end])
    longs = dict(type='scatter', name='Longs', x=np.column_stack((self.entry[self.d == 1], self.ex[self.d == 1], np.full(np.count_nonzero(self.d == 1), None))).ravel(), y=np.column_stack((self.eprice[self.d == 1], self.xprice[self.d == 1], np.full(np.count_nonzero(self.d == 1), None))).ravel(), mode='markers+lines', marker=dict(color='blue', size=6, symbol='triangle-up'), line=dict(color='blue', width=1))
    shorts = dict(type='scatter', name='Shorts', x=np.column_stack((self.entry[self.d == -1], self.ex[self.d == -1], np.full(np.count_nonzero(self.d == -1), None))).ravel(), y=np.column_stack((self.eprice[self.d == -1], self.xprice[self.d == -1], np.full(np.count_nonzero(self.d == -1), None))).ravel(), mode='markers+lines', marker=dict(color='orange', size=6, symbol='triangle-up'), line=dict(color='orange', width=1))
    # fig = go.Figure([candlesticks, longs, shorts], layout).show()
    iplot({'data':[candlesticks, longs, shorts], 'layout':layout})
    return self

  def plot_portfolio(self, shrink=True):
    i_start = 0
    i_end = self.n
    if shrink:
      i_start = np.min(self.entry[self.entry >= 0])
      i_end = np.max(self.ex[self.ex >= 0])
    layout = dict(title='Portfolio', xaxis_title='Time', yaxis_title='Value', yaxis=dict(title='Benefit', autorange=True, fixedrange=False), yaxis2=dict(title="price",overlaying="y", side="right",position=0.15))
    portfolio = dict(type='scatter', x=np.arange(i_start,i_end), y=self.val[i_start:i_end], mode='lines', line=dict(color='blue', width=1), fill='tozeroy', yaxis='y', name='Benefit')
    price = dict(type='scatter', x=np.arange(i_start,i_end), y=self.c[i_start:i_end], mode='lines', line=dict(color='orange', width=2), yaxis='y2', name='price')
    iplot({'data':[portfolio, price], 'layout':layout})
    return self
  
  def plot_concurrency(self, shrink=True):
    i_start = 0
    i_end = self.n
    if shrink:
      i_start = np.min(self.entry[self.entry >= 0])
      i_end = np.max(self.ex[self.ex >= 0])
    layout = dict(title='Concurrency', xaxis_title='Time', yaxis_title='Concurrency', yaxis=dict(autorange=True, fixedrange=False))
    concurrency = dict( type='bar', x=list(range(self.n)), y=self.num[i_start:i_end], marker=dict(color='blue'))
    iplot({'data':[concurrency], 'layout':layout})
    return self
  