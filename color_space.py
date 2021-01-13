# @yifan
# 2021.01.12
#

import numpy as np
import cv2
import os
import copy
from skimage.measure import block_reduce
from myPCA import myPCA

def YUV4202BGR(X):
    tmp = [ X[0], 
            cv2.resize(X[1], (X[0].shape[1], X[0].shape[2])),
            cv2.resize(X[2], (X[0].shape[1], X[0].shape[2]))]
    return YUV2BGR(tmp)

def BGR2RGB(X):
    R        = copy.deepcopy(X[:,:,2])
    X[:,:,2] = X[:,:,0]
    X[:,:,0] = R
    return X

def BGR2YUV(X):
    X = BGR2RGB(X)
    K = np.array([[   0.299,    0.587,    0.114],
                  [-0.14713, -0.28886,    0.436],
                  [   0.615, -0.51499, -0.10001]])
    X = np.moveaxis(X, -1, 0)
    S = X.shape
    X = np.dot(K, X.reshape(3, -1))
    X = X.reshape(S)
    X = np.moveaxis(X, 0, -1)
    return X

def YUV2BGR(X):
    K = np.array([[1,        0,  1.13983],
                  [1, -0.39465, -0.58060],
                  [1,  2.03211,        0]])
    X = np.moveaxis(X, -1, 0)
    S = X.shape
    X = np.dot(K, X.reshape(3, -1))
    X = np.moveaxis(X.reshape(S), 0, -1)
    return BGR2RGB(X)

def BGR2PQR(X):
    pca = myPCA()
    S = X.shape
    X = X.reshape(-1, 3)
    pca.fit(X)
    return pca, pca.transform(X).reshape(S)

def PQR2BGR(X, pca):
    return pca.inverse_transform(X)

def Load_YUV420_from_File(name):
    try: 
        X = cv2.imread(name)
        X.shape
    except:
        print("   <ERROR> Loading YUV420, No such file!")
        return []
    X = BGR2YUV(X)
    return [X[:,:,0], block_reduce(X[:,:,1], (2, 2), np.mean), block_reduce(X[:,:,2], (2, 2), np.mean)]

def Load_from_Folder(folder, color='PQR', ct=1):
    name = os.listdir(folder)
    name.sort()
    pqr, pca = [], []
    img = []
    Y, U, V = [], [], []
    for n in name:
        try: 
            X = cv2.imread(folder+'/'+n)
            X.shape
        except:
            continue
        if color == 'PQR':
            p, X = BGR2PQR(X)
            pqr.append(X)
            pca.append(p)
        elif color == 'BGR':
            img.append(X)
        elif color == 'YUV444':
            X = BGR2YUV(X)
            Y.append(X)
        elif color == 'YUV420':
            X = BGR2YUV(X)
            Y.append(X[:,:,0])
            U.append(block_reduce(X[:,:,1], (2, 2), np.mean))
            V.append(block_reduce(X[:,:,2], (2, 2), np.mean))
        else:
            assert (False), 'No such color type!'
        ct -= 1
        if ct == 0:
            break
    if color == 'PQR':
        return pca, pqr
    elif color == 'BGR':
        return img
    elif color == 'YUV444':
        return Y
    elif color == 'YUV420':
        return Y, U, V

    

