import numpy as np


def linearRegression(y, xStep):
  n = y.shape[1]
  x = (np.arange(n) * xStep)
  xys = x @ y.T
  x2s = x @ x.T
  xs = x @ np.ones(n)
  ys = y @ np.ones(n)  
  slopes = ((n * xys) - (xs * ys)) / ((n * x2s) - (xs * xs))
  intercepts = ((x2s * ys) - (xs * xys)) / ((n * x2s) - (xs * xs))
  return slopes, intercepts


def linearRegression_predict(y, next=1):
  xStep =1
  xNext = (y.shape[1] - 1) * xStep + (next * xStep)
  slopes, intercepts = linearRegression(y, xStep)
  return slopes * xNext + intercepts


def linearRegression_predict_upto(y, upto=1):
  xStep =1
  xNext = np.arange(y.shape[1], y.shape[1] + upto) * xStep
  xNext = xNext[:, None]
  slopes, intercepts = linearRegression(y, xStep)
  return np.transpose(slopes * xNext + intercepts)

