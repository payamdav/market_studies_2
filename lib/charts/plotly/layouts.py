

layout_price_indicator_volume = dict(
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
    domain=[0.3, 1],
    autorange=True,
    fixedrange=False,
    showspikes=True,
    spikesnap="cursor",
    spikemode="across",
    spikethickness=1,
  ),

  yaxis2=dict(
    title='y2',
    domain=[0.15, 0.27],
    autorange=True,
    fixedrange=False,
  ),

  yaxis3=dict(
    title='Volume',
    domain=[0, 0.12],
    autorange=True,
    fixedrange=False,
  )
)


layout_price_volume = dict(
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
    domain=[0.2, 1],
    autorange=True,
    fixedrange=False,
    showspikes=True,
    spikesnap="cursor",
    spikemode="across",
    spikethickness=1,
  ),

  yaxis2=dict(
    title='Volume',
    domain=[0, 0.15],
    autorange=True,
    fixedrange=False,
  )
)


layout_simple = dict(
  title='',

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
    title='',
    domain=[0, 1],
    autorange=True,
    fixedrange=False,
    showspikes=True,
    spikesnap="cursor",
    spikemode="across",
    spikethickness=1,
  ),

)

layout_simple_double_x = dict(
  title='',

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
    title='',
    domain=[0, 1],
    # autorange=True,
    # fixedrange=False,
    showspikes=True,
    spikesnap="cursor",
    spikemode="across",
    spikethickness=1,
  ),

  xaxis2=dict(
    domain=[0, 1],
    anchor= 'y',
    overlaying= 'x',
    side= 'top',
    autorange=True,
    fixedrange=False,
  )

)
