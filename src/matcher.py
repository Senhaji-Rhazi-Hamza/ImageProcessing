import numpy as np
def distances(descs1, descs2):
  m = len(descs1)
  n = len(descs2)
  a = np.array([[np.linalg.norm(descs1[i] - descs2[j]) for j in range(m)] for i in range(n)])
  return a

def getPairsMins(distances):
  a = distances.argsort()
  n = distances.shape[0]
  pairs = np.array([(i,a[i][0]) for i in range(n)])
  minns = np.array([distances[i].min() for i in range(n)])
  idx = minns.argsort()
  mins = mins[idx]
  pairs = pairs[idx]
  return pairs, mins

def matchers(kp1, des1, kp2, des2, k):
  n, m = len(des1), len(des2)
  assert(k > 0), "number of key point must be > 0"
  size = min(n,m,k)
  pairs, mins = getPairsMins(ditances(des1, des2))
  avg = mins[:k].mean()
  pairkeypoins = np.array([(kp1[pairs[i][0]],kp2[pairs[i][1]]) for i in range(len(pairs))])
  return pairs[:k], avg
