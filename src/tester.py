import numpy as np
import numpy.matlib

def distances(descs1, descs2):
    a = np.array([[np.linalg.norm(descs1[i] - descs2[j]) for j in range(descs2.shape[0])]\
            for i in range(descs1.shape[0])])
    return a

def distances2(descs1, descs2):
    n = descs1.shape[0]
    m = descs2.shape[0]
    des1rep = np.matlib.repmat(descs1,m,1)
    des2rep = np.matlib.repmat(descs2,n,1).T
    return (des1rep - des2rep)**2
def getPairsMins2(distances):
    dist = distances.argsort()
    n = distances.shape[0]
    pairs = np.array([(i,a[i][0]) for i in range(n)])
    minns = np.array([distances[i].min() for i in range(n)])
    idx = minns.argsort()
    minns = minns[idx]
    pairs = pairs[idx]
    return pairs, minns

def getPairsMins(distances):
    a = distances.argsort()
    n = distances.shape[0]
    pairs = np.array([(i,a[i][0]) for i in range(n)])
    minns = np.array([distances[i].min() for i in range(n)])
    idx = minns.argsort()
    minns = minns[idx]
    pairs = pairs[idx]
    return pairs, minns

def match2(kp1, des1, kp2, des2, k = 100):
    des1 = np.concatenate((des1[0], des1[1], des1[2], des1[3]), axis = 0)
    des2 = np.concatenate((des2[0], des2[1], des2[2], des2[3]), axis = 0)
    kp1 = np.concatenate((kp1[0], kp1[1], kp1[2], kp1[3]), axis = 0)
    kp2 = np.concatenate((kp2[0], kp2[1], kp2[2], kp2[3]), axis = 0)
    n, m = len(des1), len(des2)
    assert(k > 0), "number of key point must be > 0"
    pairs, mins = getPairsMins2(distances2(des1, des2))
    avg = mins[:k].mean()
    match = True if (avg < 1) else False
    if (match):
      print("the pictures match", "with avg ", avg)
    else:
      print("the pictures does not match","with avg ", avg) 
    return pairs[:k], avg, match

def match(kp1, des1, kp2, des2, k = 100):
    des1 = np.concatenate((des1[0], des1[1], des1[2], des1[3]), axis = 0)
    des2 = np.concatenate((des2[0], des2[1], des2[2], des2[3]), axis = 0)
    kp1 = np.concatenate((kp1[0], kp1[1], kp1[2], kp1[3]), axis = 0)
    kp2 = np.concatenate((kp2[0], kp2[1], kp2[2], kp2[3]), axis = 0)
    n, m = len(des1), len(des2)
    assert(k > 0), "number of key point must be > 0"
    size = min(n,m,k)
    pairs, mins = getPairsMins(distances(des1, des2))
    avg = mins[:k].mean()
    match = True if (avg < 1) else False
    if (match):
      print("the pictures match", "with avg ", avg)
    else:
      print("the pictures does not match","with avg ", avg)
    return pairs[:k], avg, match
