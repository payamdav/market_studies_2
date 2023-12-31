from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import lib.pandas.pandas_extension
from lib.indicators.levelizer import levelizer_by_absolute_threshold, levelizer_by_absolute_threshold_adjust

layout = dict(
  title='Candles',

  xaxis=dict(
    title='Index',
    domain=[0, 1],
    showspikes=True,
    spikesnap="cursor",
    spikemode="across",
    spikethickness=1,

    rangeslider=dict(
      visible=False,
      thickness=0.02,
    )
  ),

  yaxis=dict(
    title='Price',
    domain=[0.15, 1],
    autorange=True,
    fixedrange=False,
    showspikes=True,
    spikesnap="cursor",
    spikemode="across",
    spikethickness=1,
  ),

  yaxis2=dict(
    title='Volume',
    domain=[0, 0.12],
    autorange=True,
    fixedrange=False,
  )

)

traces = []

def main():
  df = pd.DataFrame().ext.load('EURUSD')
  df = df.ext.resample_candles('1T')
  df = df[-10000:]
  df.reset_index(inplace=True, drop=True)

  timeperiod_check = 240

  maxv = df.v.max()

  p = (df.h + df.l + df.c) / 3
  l1 = levelizer_by_absolute_threshold(p, df.ext.atr(14) * 1.5)
  ema1 = p.ext.ema(8)
  ema2 = p.ext.ema(21)
  atr = df.ext.atr(14)

  s1 = ema1 - atr
  s2 = ema2 - 2 * atr
  r1 = ema1 + atr
  r2 = ema2 + 2 * atr


  traces.append(dict(type='candlestick', name='Candles', x=df.index, open=df.o, high=df.h, low=df.l, close=df.c, yaxis='y1'))
  traces.append(dict(type='scatter', name='p', x=df.index, y=p, yaxis='y1'))
  traces.append(dict(type='scatter', name='ema1', x=df.index, y=ema1, yaxis='y1'))
  traces.append(dict(type='scatter', name='ema2', x=df.index, y=ema2, yaxis='y1'))
  traces.append(dict(type='scatter', name='l1', x=df.index, y=l1, yaxis='y1'))

  traces.append(dict(type='scatter', name='s1', x=df.index, y=s1, yaxis='y1'))
  traces.append(dict(type='scatter', name='s2', x=df.index, y=s2, yaxis='y1'))
  traces.append(dict(type='scatter', name='r1', x=df.index, y=r1, yaxis='y1'))
  traces.append(dict(type='scatter', name='r2', x=df.index, y=r2, yaxis='y1'))



  traces.append(dict(type='bar', name='Volume', x=df.index, y=df.v, yaxis='y2'))
  # traces.append(dict(type='scatter', name='Volume3',  x=df.index,  y=df.v.ext.ema(14),  yaxis='y2'))

  
  fig = go.Figure(traces, layout).show()


main()
