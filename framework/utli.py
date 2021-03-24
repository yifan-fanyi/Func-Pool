# 2020.01.27
# @yifan
# ulti.py
#
import numpy as np
import copy
import math
from framework.dependency import *

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

def load_img(img, isYUV=False):
    if isYUV == True:
        pqr = Load_Images(['/Users/alex/Desktop/proj/compression/data/DIV2K/DIV2K/'+str(img)+'.bmp'],color='YUV')
    else:
        pca, pqr = Load_Images(['/Users/alex/Desktop/proj/compression/data/DIV2K/DIV2K/'+str(img)+'.bmp'],color='PQR')
    Y = pqr[0][:,:,0].reshape(-1,pqr[0].shape[0],pqr[0].shape[1],1)[:, :1024, :1024 ]
    return Y

def load_img_Kodak(img, isYUV=False):
    if isYUV == True:
        pqr = Load_Images(['/Users/alex/Desktop/proj/compression/data/Kodak/Kodak/'+str(img)+'.bmp'],color='YUV')
    else:
        pca, pqr = Load_Images(['/Users/alex/Desktop/proj/compression/data/Kodak/Kodak/'+str(img)+'.bmp'],color='PQR')
    Y = pqr[0][:,:,0].reshape(-1,pqr[0].shape[0],pqr[0].shape[1],1)
    U = pqr[0][:,:,1].reshape(-1,pqr[0].shape[0],pqr[0].shape[1],1)
    V = pqr[0][:,:,2].reshape(-1,pqr[0].shape[0],pqr[0].shape[1],1)
    return Y, U, V