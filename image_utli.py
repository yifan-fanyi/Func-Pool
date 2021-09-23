# 20201.03.18
import numpy as np
from skimage.measure import block_reduce
import cv2

def MeanPooling(X, win=2):
    return block_reduce(X, (1, win, win, 1), np.mean)

def MaxPooling(X, win=2):
    return block_reduce(X, (1, win, win, 1), np.max)

def mybilinear_interpolation(X, win):
    eX = np.zeros((X.shape[0], X.shape[1]*win, X.shape[2]*win, X.shape[-1]))
    for i in range(eX.shape[1]):
        for j in range(eX.shape[2]):
            ii = (float)(i % win) / (float)(win)
            iii = i // win
            jj = (float)(j % win) / (float)(win)
            jjj = j // win
            a = X[:, iii,   jjj]
            if iii+1 < X.shape[1]:
                b = X[:, iii+1, jjj]
                if jjj+1 < X.shape[2]:
                    c = X[:, iii,   jjj+1]
                    d = X[:, iii+1, jjj+1]
                else:
                    c = X[:, iii,   jjj]
                    d = X[:, iii+1, jjj]
            else:
                b = X[:, iii,   jjj ]
                if jjj+1 < X.shape[2]:
                    c = X[:, iii,   jjj+1]
                    d = X[:, iii, jjj+1]
                else:
                    c = X[:, iii,   jjj]
                    d = X[:, iii, jjj]
            eX[:, i, j] =  a  * (1-ii) * (1-jj) + \
                           b  * (ii)   * (1-jj) + \
                           c  * (1-ii) * (jj)   + \
                           d  * (ii)   * (jj) 
    return eX

def interpolation(X, win):
    eX = np.zeros((X.shape[0], X.shape[1]*win, X.shape[2]*win, X.shape[-1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[-1]):
            eX[i,:,:,j] = cv2.resize(X[i,:,:,j], (X.shape[2]*win, X.shape[1]*win))
    return eX