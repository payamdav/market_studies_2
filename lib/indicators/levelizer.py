import numpy as np

def levelizer_by_absolute_threshold_adjust(data, threshold):
  l = np.zeros(len(data))
  t = np.broadcast_to(threshold, len(data))
  l[0] = data[0]
  levelStart = 0
  for i in range(1, len(data)):
    if data[i] >= l[i-1] + t[i]:
      levelStart = i
      l[i] = data[i]
    elif data[i] <= l[i-1] - t[i]:
      levelStart = i
      l[i] = data[i]
    else:
      levelValue = data[levelStart:i+1].mean()
      l[levelStart:i+1] = levelValue
  return l

def levelizer_by_absolute_threshold(data, threshold):
  l = np.zeros(len(data))
  t = np.broadcast_to(threshold, len(data))
  l[0] = data[0]
  for i in range(1, len(data)):
    if data[i] >= l[i-1] + t[i]:
      l[i] = data[i]
    elif data[i] <= l[i-1] - t[i]:
      l[i] = data[i]
    else:
      l[i] = l[i-1]
  return l

def levelizer_by_absolute_threshold_2(data, threshold):
  zz = np.zeros(len(data))
  zz[0] = data[0]
  for i in range(1, len(data)):
    if zz[i-1] == 0:
      zz[i] = data[i]
    else:
      if data[i] >= zz[i-1] + threshold:
        zz[i] = data[i]
      elif data[i] <= zz[i-1] - threshold:
        zz[i] = data[i]
      else:
        zz[i] = zz[i-1]
  return zz

def levelizer_by_deviation(data, deviation=0.0005):
  zz = np.zeros(len(data))
  zz[0] = data[0]
  for i in range(1, len(data)):
    if zz[i-1] == 0:
      zz[i] = data[i]
    else:
      if data[i] >= zz[i-1] * (1 + deviation):
        zz[i] = data[i]
      elif data[i] <= zz[i-1] * (1 - deviation):
        zz[i] = data[i]
      else:
        zz[i] = zz[i-1]
  return zz
