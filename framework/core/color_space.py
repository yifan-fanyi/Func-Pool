# 2020.01.28
# @yifan
# color space conversion
#
import numpy as np
import copy
from sklearn.preprocessing import normalize

from framework.core.myPCA import myPCA
from framework.core.llsr import LLSR

def Clip(X):
    tmp = copy.deepcopy(X)
    tmp = tmp.astype('int16')
    tmp[tmp > 255] = 255
    tmp[tmp < 0] = 0
    return tmp

def YUV4202BGR(X):
    tmp = [ X[0], 
            cv2.resize(X[1], (X[0].shape[1], X[0].shape[2])),
            cv2.resize(X[2], (X[0].shape[1], X[0].shape[2]))]
    return YUV2BGR(tmp)
    
def BGR2PQR(X):
    def reScale(K):
        K[:1] = normalize(K[:1], norm='l1')
        K[0,0] *= 219/255
        K[0,1] *= 219/255
        K[0,2] *= 219/255
        sb = 224/255/(np.sum(np.abs(K[1])))
        K[1] *= sb
        sc = 224/255/(np.sum(np.abs(K[2])))
        K[2] *= sc
        return K
    pca = myPCA()
    S = X.shape
    X = X.reshape(-1, 3)
    pca.fit(X)
    pca.Kernels = reScale(pca.Kernels)
    return pca, pca.transform(X).reshape(S)

def PQR2BGR(X, pca):
    return pca.inverse_transform(X, K=np.linalg.inv(pca.Kernels))

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
    return Clip(BGR2RGB(X))

def ML_inv_color(X_bgr, iX):
    llsr = LLSR(onehot=False)
    llsr.fit(iX.reshape(-1,3), X_bgr.reshape(-1,3))
    iX = llsr.predict_proba(iX.reshape(-1,3)).reshape(X_bgr.shape)
    iX = Clip(iX.astype('int32'))
    return iX
