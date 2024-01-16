import numpy as np

def lr_slope(y: np.ndarray, x: np.ndarray) -> np.ndarray:
  """
  Calculate the slope of a linear regression line for each row of y and x.
  """
  if y.ndim == 1:
    y = y[:, None]
  if x.ndim == 1:
    x = x[:, None]
  if y.shape[0] != x.shape[0]:
    raise ValueError('x and y must have the same number of rows')
  if y.shape[0] == 1:
    return np.nan
  x_mean = np.mean(x, axis=1)[:, None]
  y_mean = np.mean(y, axis=1)[:, None]
  num = np.sum((x - x_mean) * (y - y_mean), axis=1)[:, None]
  den = np.sum((x - x_mean) ** 2, axis=1)[:, None]
  slopes = num / den
  # calculate degrees of slopes
  degrees = np.degrees(np.arctan(slopes))
  sines = np.sin(np.radians(degrees))
  return slopes, degrees, sines

def labelBySlope(y, period):
  if not isinstance(y, np.ndarray):
    y = y.to_numpy()
  idx = np.arange(y.shape[0])[:, None] + np.arange(1, period + 1, 1)[None, :]
  idx = np.clip(idx, 0, y.shape[0] - 1)
  yy = y[idx]
  yydm = (np.mean(np.abs(np.diff(yy, axis=1)), axis=1))[:, None]
  yydm = np.clip(yydm, 0.000001, None)
  x = yydm * np.arange(period)[None, :]
  slopes, degrees, sines = lr_slope(yy, x)
  # print(slopes.ravel())
  # print(degrees.ravel())
  return slopes.ravel(), degrees.ravel(), sines.ravel()






def main():
  y = np.array([1,2,3,4,5,6,7,8,9,10])
  # y = np.array([1,-2,-3,-4,5,-6,-7,8,9,10])
  print(y)
  print('----------')
  labelBySlope(y, 3)


if __name__ == '__main__':
  main()


  
