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

  timeperiod_check = 240

  maxv = df.v.max()

  atr = df.ext.atr(14)
  p_last = ((df.h + df.l + df.c) / 3).ext.ma(timeperiod_check).shift(1)
  h_last = df.h.ext.ma(timeperiod_check).shift(1)
  l_last = df.l.ext.ma(timeperiod_check).shift(1)
  # r1 = 2 * p_last - l_last
  # r2 = p_last + h_last - l_last
  # r3 = r1 + h_last - l_last
  # s1 = 2 * p_last - h_last
  # s2 = p_last - h_last + l_last
  # s3 = s1 - h_last + l_last

  s1 = p_last - (h_last - p_last)
  s2 = p_last - 2 * (h_last - p_last)
  s3 = p_last - 3 * (h_last - p_last)
  s4 = p_last - 4 * (h_last - p_last)
  s5 = p_last - 5 * (h_last - p_last)
  r1 = p_last + (p_last - l_last)
  r2 = p_last + 2 * (p_last - l_last)
  r3 = p_last + 3 * (p_last - l_last)
  r4 = p_last + 4 * (p_last - l_last)
  r5 = p_last + 5 * (p_last - l_last)

  long_s1 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) > df.c) & (df.c > s1) & (df.l < s1) & (df.c < df.o) & ((df.o - df.c) < (df.c - df.l))
  long_s2 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) > df.c) & (df.c > s2) & (df.l < s2) & (df.c < df.o) & ((df.o - df.c) < (df.c - df.l))
  long_s3 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) > df.c) & (df.c > s3) & (df.l < s3) & (df.c < df.o) & ((df.o - df.c) < (df.c - df.l))
  long_s4 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) > df.c) & (df.c > s4) & (df.l < s4) & (df.c < df.o) & ((df.o - df.c) < (df.c - df.l))
  long_s5 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) > df.c) & (df.c > s5) & (df.l < s5) & (df.c < df.o) & ((df.o - df.c) < (df.c - df.l))
  short_r1 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) < df.c) & (df.c < r1) & (df.h > r1) & (df.c > df.o) & ((df.c - df.o) < (df.h - df.c))
  short_r2 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) < df.c) & (df.c < r2) & (df.h > r2) & (df.c > df.o) & ((df.c - df.o) < (df.h - df.c))
  short_r3 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) < df.c) & (df.c < r3) & (df.h > r3) & (df.c > df.o) & ((df.c - df.o) < (df.h - df.c))
  short_r4 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) < df.c) & (df.c < r4) & (df.h > r4) & (df.c > df.o) & ((df.c - df.o) < (df.h - df.c))
  short_r5 = (df.v > df.v.ext.ma(10)) & (df.v > 0) & (df.c.shift(1) < df.c) & (df.c < r5) & (df.h > r5) & (df.c > df.o) & ((df.c - df.o) < (df.h - df.c))

  print(f'long s1 count: {np.count_nonzero(long_s1)}')
  print(f'short r1 count: {np.count_nonzero(short_r1)}')
  print(f'long s5 count: {np.count_nonzero(long_s5)}')
  print(f'short r5 count: {np.count_nonzero(short_r5)}')

  print(f'l_s_1: {np.count_nonzero(long_s1)} - l_s_2: {np.count_nonzero(long_s2)} - l_s_3: {np.count_nonzero(long_s3)} - l_s_4: {np.count_nonzero(long_s4)} - l_s_5: {np.count_nonzero(long_s5)}')
  print(f's_r_1: {np.count_nonzero(short_r1)} - s_r_2: {np.count_nonzero(short_r2)} - s_r_3: {np.count_nonzero(short_r3)} - s_r_4: {np.count_nonzero(short_r4)} - s_r_5: {np.count_nonzero(short_r5)}')
  
  traces.append(dict(type='candlestick', name='Candles', x=df.index, open=df.o, high=df.h, low=df.l, close=df.c, yaxis='y1'))
  traces.append(dict(type='scatter', name='s1', x=df.index, y=s1, yaxis='y1'))
  traces.append(dict(type='scatter', name='s2', x=df.index, y=s2, yaxis='y1'))
  traces.append(dict(type='scatter', name='s3', x=df.index, y=s3, yaxis='y1'))
  traces.append(dict(type='scatter', name='s4', x=df.index, y=s4, yaxis='y1'))
  traces.append(dict(type='scatter', name='s5', x=df.index, y=s5, yaxis='y1'))
  traces.append(dict(type='scatter', name='r1', x=df.index, y=r1, yaxis='y1'))
  traces.append(dict(type='scatter', name='r2', x=df.index, y=r2, yaxis='y1'))
  traces.append(dict(type='scatter', name='r3', x=df.index, y=r3, yaxis='y1'))
  traces.append(dict(type='scatter', name='r4', x=df.index, y=r4, yaxis='y1'))
  traces.append(dict(type='scatter', name='r5', x=df.index, y=r5, yaxis='y1'))
  # traces.append(dict(type='scatter', name='ema8', x=df.index, y=p.ext.ema(8), yaxis='y1'))
  # traces.append(dict(type='scatter', name='ema21', x=df.index, y=p.ext.ema(21), yaxis='y1'))



  traces.append(dict(type='bar', name='Volume', x=df.index, y=df.v, yaxis='y2'))
  # traces.append(dict(type='scatter', name='Volume3',  x=df.index,  y=df.v.ext.ema(14),  yaxis='y2'))
  traces.append(dict(type='scatter', name='long_s1', x=df.index, y=maxv * long_s1.mask(long_s1 == False), mode='markers' ,yaxis='y2'))
  traces.append(dict(type='scatter', name='long_s2', x=df.index, y=maxv * long_s2.mask(long_s2 == False), mode='markers' ,yaxis='y2'))
  traces.append(dict(type='scatter', name='long_s3', x=df.index, y=maxv * long_s3.mask(long_s3 == False), mode='markers' ,yaxis='y2'))
  traces.append(dict(type='scatter', name='long_s4', x=df.index, y=maxv * long_s4.mask(long_s4 == False), mode='markers' ,yaxis='y2'))
  traces.append(dict(type='scatter', name='long_s5', x=df.index, y=maxv * long_s5.mask(long_s5 == False), mode='markers' ,yaxis='y2'))
  traces.append(dict(type='scatter', name='short_r1', x=df.index, y=maxv * short_r1.mask(short_r1 == False), mode='markers' ,yaxis='y2'))
  traces.append(dict(type='scatter', name='short_r2', x=df.index, y=maxv * short_r2.mask(short_r2 == False), mode='markers' ,yaxis='y2'))
  traces.append(dict(type='scatter', name='short_r3', x=df.index, y=maxv * short_r3.mask(short_r3 == False), mode='markers' ,yaxis='y2'))
  traces.append(dict(type='scatter', name='short_r4', x=df.index, y=maxv * short_r4.mask(short_r4 == False), mode='markers' ,yaxis='y2'))  
  traces.append(dict(type='scatter', name='short_r5', x=df.index, y=maxv * short_r5.mask(short_r5 == False), mode='markers' ,yaxis='y2'))

  
  fig = go.Figure(traces, layout).show()


main()
