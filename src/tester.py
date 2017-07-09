import numpy as np

def distances(descs1, descs2):
    m = len(descs1)
    n = len(descs2)
    a = np.array([[np.linalg.norm(descs1[i] - descs2[j]) for j in range(m)]\
            for i in range(n)])
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

def match(kp1, des1, kp2, des2, k = 10):
    n, m = len(des1), len(des2)
    assert(k > 0), "number of key point must be > 0"
    size = min(n,m,k)
    pairs, mins = getPairsMins(ditances(descs1, descs2))
    avg = mins[:k].mean()
    return pairs[:k], avg


