# 2021.01.27
# @yifan 
#
import numpy as np
import copy
from skimage.util import view_as_windows
from framework.core.LANCZOS import LANCZOS

def Shrink(X, win):
    X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def invShrink(X, win):
    S = X.shape
    X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
    X = np.moveaxis(X, 5, 2)
    X = np.moveaxis(X, 6, 4)
    return X.reshape(S[0], win*S[1], win*S[2], -1)

def DownSample(X, r):
    a, b =  LANCZOS.split(X, r)   
    return a

def Hist(X, bins=-1):
    if bins < 0:
        bins = (int)(2* np.max([np.max(X), np.abs(np.min(X))])+7)
    X = np.round(X.reshape(-1)).astype('int32')
    hist, _ = np.histogram(X, bins=bins)
    return np.arange(-(bins//2), bins//2+1, 1), hist/len(X)

def mySort(abc):
    def sortbyAxis(abc, axis):
        sorted_abc = []
        for i in range(len(abc)):
            tmp = []
            idx = np.argmin(abc[:,axis])
            sorted_abc.append(abc[idx])
            abc = np.delete(abc, idx, 0)
        return np.array(sorted_abc)
    for i in range(abc.shape[1]):
        if i > 0:
            start = 0
            for j in range(1, len(abc)):
                if abs(abc[start, i-1] - abc[j, i-1]) > 1e-10:
                    abc[start:j] = sortbyAxis(abc[start:j], i)
                    start = j 
                if j == len(abc)-1:
                    abc[start:j+1] = sortbyAxis(abc[start:j+1], i)
        else:
            abc = sortbyAxis(abc, 0)
    return abc  

def write_to_txt(X, name='tmp.txt'):
    X = copy.deepcopy(X)
    X = X.astype('int32')
    ct = 0
    with open(name, 'a') as f:
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                for k in range(X.shape[3]):
                    ct += 1
                    f.write(str(X[0,i,j,k]))
                    f.write('\n')
    print('write ',ct, 'to',name, np.max(X), np.min(X))

def n_zero(X, percent=True):
    n = 0
    tX = copy.deepcopy(X).astype('int16')
    tX[tX != 0] = 1
    if percent == True:
        zp = 1-np.sum(tX)/(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3])
    else:
        zp = (X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]) - np.sum(tX)
    return zp

def Distortion_model(x, a, b):
    return a * np.power(x, b)
