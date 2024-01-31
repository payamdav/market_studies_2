class Params:
  def __init__(self, default=None) -> None:
    self.default = default

  def __getattr__(self, name):
    if name == 'default':
      return self.default
    elif self.default is not None:
      return getattr(self.default, name)
    else:
      raise AttributeError(f'Params has no attribute {name}')
  
  @property
  def week_minutes(self):
    return 7 * 24 * 60 if self.market == 'crypto' else 5 * 24 * 60


default = Params()
default.symbol = 'EURUSD'
default.market = 'forex'
default.slicer = slice(-20000, None)
default.win = 100
default.win_level = 10000
default.plot_before = 1000
default.plot_after = 1000
default.plot_each = True
default.plot = True
default.precision = 5

eurusd = Params(default)
eurusd.symbol = 'EURUSD'


btcusdt = Params(default)
btcusdt.symbol = 'BTCUSDT'
btcusdt.market = 'crypto'
btcusdt.precision = -1

btcusdt.default.testatr = 12





