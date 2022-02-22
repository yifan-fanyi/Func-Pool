# 2021.01.27
# @yifan 
#
import numpy as np
from scipy.fftpack import dct, idct

from LLSR import LLSR
from ZigZag import ZigZag

class DCT():
    def __init__(self, N=8, P=8):
        self.N = N
        self.P = P
        self.W = 8
        self.H = 8
    
    def transform(self, a):
        S = list(a.shape)
        a = a.reshape(-1, self.N, self.P, 1)
        a = dct(dct(a, axis=1, norm='ortho'), axis=2, norm='ortho')
        return a.reshape(S)

    def inverse_transform(self, a):
        S = list(a.shape)
        a = a.reshape(-1, self.N, self.P, 1)
        a = idct(idct(a, axis=1, norm='ortho'), axis=2, norm='ortho')
        return a.reshape(S)
    
    def ML_inverse_transform(self, Xraw, X):
        llsr = LLSR(onehot=False)
        llsr.fit(X.reshape(-1, X.shape[-1]), Xraw.reshape(-1, X.shape[-1]))
        S = X.shape
        X = llsr.predict_proba(X.reshape(-1, X.shape[-1])).reshape(S)
        return X


class DCT3D():
    def __init__(self, N=8, P=8, C=3, ):
        self.N = N
        self.P = P
        self.C = C
    
    def fit(self, X):
        return self

    def transform(self, a):
        S = list(a.shape)
        a = a.reshape(-1, self.C, self.N, self.P)
        a = dct(a, axis=1, norm='ortho')
        a = dct(a, axis=2, norm='ortho')
        a = dct(a, axis=3, norm='ortho')
        return a.reshape(S)

    def inverse_transform(self, a):
        S = list(a.shape)
        a = a.reshape(-1, self.C, self.N, self.P)
        a = idct(a, axis=1, norm='ortho')
        a = idct(a, axis=2, norm='ortho')
        a = idct(a, axis=3, norm='ortho')
        return a.reshape(S)

# use DCT and IDCT for interpolation 
# input for Smooth_Interpolation: (?, initN, initN, 1)
# return for Smooth_Interpolation: (?, targetN, targetN, 1)
class Smooth_Interpolation():
    def __init__(self, initN=8, targetN=16, mode='block'):
        self.initN = initN
        self.targetN = targetN
        self.mode = mode
        self.dct1 = DCT(N=initN, P=initN)
        self.dct2 = DCT(N=targetN, P=targetN)
        
    def add_zeros_zigzag(self, X):
        zigzag1 = ZigZag(N=self.initN)
        zigzag2 = ZigZag(N=self.targetN)
        X = zigzag1.transform(X).reshape(X.shape[0], -1)
        X = np.concatenate((X, np.zeros((X.shape[0], self.targetN**2 - self.initN**2))), axis=1)
        X = zigzag2.inverse_transform(X).reshape(X.shape[0], self.targetN, self.targetN, 1)
        return X

    def add_zeros_block(self, X):
        X = np.concatenate((X, np.zeros((X.shape[0], self.targetN - self.initN, X.shape[2], 1))), axis=1)
        X = np.concatenate((X, np.zeros((X.shape[0], X.shape[1], self.targetN - self.initN, 1))), axis=2)
        return X

    def remove_zigzag_(self, X):
        zigzag1 = ZigZag(N=self.initN)
        zigzag2 = ZigZag(N=self.targetN)
        X = zigzag1.transform(X).reshape(X.shape[0], -1)
        X = X[:, :self.targetN**2]
        X = zigzag2.inverse_transform(X).reshape(X.shape[0], self.targetN, self.targetN, 1)
        return X

    def remove_block_(self, X):
        X = X[:, :self.targetN, :self.targetN]
        return X

    def rescale_(self, X):
        X *= self.targetN / self.initN
        return X

    def transform(self, X):
        assert (X.shape[1] ==  X.shape[2]), "<Error> Input shape not match!"
        X = self.dct1.transform(X)
        if self.initN < self.targetN:
            if self.mode == 'zigzag':
                X = self.add_zeros_zigzag(X)
            elif self.mode == 'block':
                X = self.add_zeros_block(X)
        elif self.initN > self.targetN:
            if self.mode == 'zigzag':
                X = self.remove_zigzag_(X)
            elif self.mode == 'block':
                X = self.remove_block_(X)
        X = self.rescale_(X)
        X = self.dct2.inverse_transform(X)
        return X.reshape(X.shape[0], self.targetN, self.targetN, 1)