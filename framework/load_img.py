# @yifan
# 2021.01.12
#

import numpy as np
import cv2
import os
from skimage.measure import block_reduce

from framework.core.color_space import YUV4202BGR, BGR2RGB, BGR2YUV, YUV2BGR, BGR2PQR, PQR2BGR, ML_inv_color

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
