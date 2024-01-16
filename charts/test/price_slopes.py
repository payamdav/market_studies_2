import pandas as pd
import numpy as np
import lib.pandas.pandas_extension
import plotly.graph_objects as go
from lib.charts.plotly.layouts import layout_price_indicator_volume

def main():
  df = pd.DataFrame().ext.load('EURUSD')
  df = df.iloc[-3000:]
  p = (df.h + df.l + df.c) / 3
  slopes = p.ext.labeler_by_slope(60)
  degrees = p.ext.labeler_by_slope_degrees(60)
  sines = p.ext.labeler_by_slope_sines(30)

  traces = []
  traces.append(dict(type='scatter', name='p', x=p.index, y=p, yaxis='y1', mode='lines'))
  # traces.append(dict(type='scatter', name='slopes', x=p.index, y=slopes, yaxis='y2', mode='lines'))
  # traces.append(dict(type='scatter', name='slopes', x=p.index, y=degrees, yaxis='y2', mode='lines'))
  traces.append(dict(type='scatter', name='slopes', x=p.index, y=sines, yaxis='y2', mode='lines'))
  go.Figure(traces, layout_price_indicator_volume).show()


main()

