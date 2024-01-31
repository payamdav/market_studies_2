import numpy as np

def zigzag(h: np.ndarray, l:np.ndarray, d:np.ndarray | float) -> (np.ndarray, np.ndarray):
  """
  ZigZag indicator
  """
  if not isinstance(d, np.ndarray):
    d = np.full(len(h), d)
  oh = np.zeros(len(h), dtype=bool)
  ol = np.zeros(len(h), dtype=bool)

  start_point = np.argmax(h)
  oh[start_point] = True
  lastType = 1
  lastValue = h[start_point]
  delta = d[start_point]
  nextValue = None
  nextIndex = None

  for i in range(start_point+1, len(h)):
    if lastType == 1:
      if (l[i] < lastValue - delta and nextIndex is None) or (nextIndex is not None and l[i] < nextValue):
        nextValue = l[i]
        nextIndex = i
      elif nextIndex is not None and h[i] > nextValue + delta:
        ol[nextIndex] = True
        lastValue = nextValue
        lastType = -1
        nextIndex = i
        nextValue = h[i]
        delta = d[i]
    elif lastType == -1:
      if (h[i] > lastValue + delta and nextIndex is None) or (nextIndex is not None and h[i] > nextValue):
        nextValue = h[i]
        nextIndex = i
      elif nextIndex is not None and l[i] < nextValue - delta:
        oh[nextIndex] = True
        lastValue = nextValue
        lastType = 1
        nextIndex = i
        nextValue = l[i]
        delta = d[i]
  if nextIndex is not None:
    if lastType == 1:
      ol[nextIndex] = True
    else:
      oh[nextIndex] = True
  
  lastType = 1
  lastValue = h[start_point]
  delta = d[start_point]
  nextValue = None
  nextIndex = None
  for i in range(start_point-1, -1, -1):
    if lastType == 1:
      if (l[i] < lastValue - delta and nextIndex is None) or (nextIndex is not None and l[i] < nextValue):
        nextValue = l[i]
        nextIndex = i
      elif nextIndex is not None and h[i] > nextValue + delta:
        ol[nextIndex] = True
        lastValue = nextValue
        lastType = -1
        nextIndex = i
        nextValue = h[i]
        delta = d[i]
    elif lastType == -1:
      if (h[i] > lastValue + delta and nextIndex is None) or (nextIndex is not None and h[i] > nextValue):
        nextValue = h[i]
        nextIndex = i
      elif nextIndex is not None and l[i] < nextValue - delta:
        oh[nextIndex] = True
        lastValue = nextValue
        lastType = 1
        nextIndex = i
        nextValue = l[i]
        delta = d[i]
  if nextIndex is not None:
    if lastType == 1:
      ol[nextIndex] = True
    else:
      oh[nextIndex] = True
  return oh, ol
