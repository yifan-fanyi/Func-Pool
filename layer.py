# v2020.03.28
import numpy as np 
import cv2
from skimage.measure import block_reduce

def myResize(x, H, W):
    new_x = np.zeros((x.shape[0], H, W, x.shape[3]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[3]):
            new_x[i,:,:,j] = cv2.resize(x[i,:,:,j], (W,H), interpolation=cv2.INTER_CUBIC)
    return new_x

def MaxPooling(x):
    return block_reduce(x, (1, 2, 2, 1), np.max)

def AvgPooling(x):
    return block_reduce(x, (1, 2, 2, 1), np.mean)

def Project_concat(feature):
    dim = 0
    for i in range(0, len(feature)):
        dim += feature[i].shape[-1]
        feature[i] = np.moveaxis(feature[i],0,2)
    result = np.zeros((feature[0].shape[0],feature[0].shape[1],feature[0].shape[2],dim))
    for i in range(0,feature[0].shape[0]):
        for j in range(0,feature[0].shape[1]):
            last = feature[0].shape[0]
            tmp = []
            for fea in feature:
                scale = last/ fea.shape[0]
                tmp.append(fea[int(i/scale),int(j/scale)])
            result[i,j] = np.concatenate(tmp, axis=1)
    result = np.moveaxis(result, 2, 0)
    return result

from skimage.util import view_as_windows
# view as window and its inverse
# win = stride
def View_As_Win(X, win):
    X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def invShrink(X, win):
    S = X.shape
    X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
    X = np.moveaxis(X, 5, 2)
    X = np.moveaxis(X, 6, 4)
    return X.reshape(S[0], win*S[1], win*S[2], -1)