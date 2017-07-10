import numpy as np
from time import time

def distances(descs1, descs2):
    t = time()
    
    # a = np.array([[np.linalg.norm(descs1[i] - descs2[j]) for j in range(descs2.shape[0])]\
    #        for i in range(descs1.shape[0])])
    print('distances: {:.2f}s'.format(time() - t))
    return a

def getPairsMins(distances):
    t = time()
    a = distances.argsort()
    print('first argsort: {:.2f}s'.format(time() - t))
    n = distances.shape[0]
    pairs = np.array([(i,a[i][0]) for i in range(n)])
    minns = np.array([distances[i].min() for i in range(n)])
    t = time()
    idx = minns.argsort()
    print('second argsort: {:.2f}s'.format(time() - t))
    minns = minns[idx]
    pairs = pairs[idx]
    return pairs, minns

def match(kp1, des1, kp2, des2, k = 100):
    assert(k > 0), "number of key point must be > 0"
    t = time()
    des1 = np.concatenate((des1[0], des1[1], des1[2], des1[3]), axis = 0)
    des2 = np.concatenate((des2[0], des2[1], des2[2], des2[3]), axis = 0)
    kp1 = np.concatenate((kp1[0], kp1[1], kp1[2], kp1[3]), axis = 0)
    kp2 = np.concatenate((kp2[0], kp2[1], kp2[2], kp2[3]), axis = 0)
    print('concatenation of octaves: {:.2f}s'.format(time() - t))
    n, m = len(des1), len(des2)
    size = min(n,m,k)
    t = time()
    pairs, mins = getPairsMins(distances(des1, des2))
    print('pairmins bullshit: {:.2f}s'.format(time() - t))
    avg = mins[:k].mean()
    match = True if (avg < 1) else False
    if (match):
      print("the pictures match", "with avg ", avg)
    else:
      print("the pictures does not match","with avg ", avg)
    return pairs[:k], avg, match
