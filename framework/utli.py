# 2020.01.27
# @yifan
# ulti.py
#
import numpy as np
import copy

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