from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import lib.pandas.pandas_extension

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
  df = df.ext.resample_candles('5T')
  df = df[-1000:]
  df.reset_index(inplace=True, drop=True)

  p = (df.h + df.l + df.c) / 3
  maxp = df.h.max()
  maxv = df.v.max()
  upper = np.where(df.o > df.c, df.o, df.c)
  lower = np.where(df.o < df.c, df.o, df.c)
  color = np.where(df.o > df.c, 'red', 'green')
  pinball = ((df.h - upper) > (3 * (upper - lower))) & (3 * (df.l - lower) < (df.h - upper))
  highvol = df.v > (1.5 * df.v.ext.ma(30))
  upper_under = upper < p.ext.ema(21)
  high_over = df.h > p.ext.ema(21)
  signals = pinball & highvol & upper_under & high_over
  print(f'signal count: {np.count_nonzero(signals)}')


  traces.append(dict(type='candlestick', name='Candles', x=df.index, open=df.o, high=df.h, low=df.l, close=df.c, yaxis='y1'))
  traces.append(dict(type='scatter', name='ema8', x=df.index, y=p.ext.ema(8), yaxis='y1'))
  traces.append(dict(type='scatter', name='ema21', x=df.index, y=p.ext.ema(21), yaxis='y1'))

  traces.append(dict(type='scatter', name='signals', x=df.index, y=maxv * signals.mask(signals == False), mode='markers' ,yaxis='y2'))


  traces.append(dict(type='bar', name='Volume', x=df.index, y=df.v, yaxis='y2'))
  traces.append(dict(type='scatter', name='Volume3',  x=df.index,  y=df.v.ext.ema(14),  yaxis='y2'))
  
  fig = go.Figure(traces, layout).show()


main()
