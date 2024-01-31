import numpy as np
from scipy.signal import find_peaks, windows


def merge_values_by_kernel(v: np.ndarray, weights: [np.ndarray], precision: float, distance: float) -> np.ndarray:
  if weights is None:
    weights = [np.ones(len(v), dtype=float)]
  vmax = v.max()
  vmin = v.min()
  vx = np.arange(vmin - precision, vmax + 2 * precision, precision)
  vy = np.zeros(len(vx))
  w = np.ones(len(v), dtype=float)
  for i in range(len(weights)):
    w = w * weights[i]
  w = w / w.sum()
  for i in range(len(v)):
    vy[np.argmin(np.abs(vx - v[i]))] += w[i]
  vy = vy / vy.sum()
  kernel = windows.gaussian(int(distance / precision), 1 / precision)
  vy = np.convolve(vy, kernel, mode='same')
  vy = vy / vy.sum()
  peaks, _ = find_peaks(vy, distance=int(distance / precision))
  return np.sort(vx[peaks])



