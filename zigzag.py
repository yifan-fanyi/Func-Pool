# 2020.01.27
# @yifan
# Zigzag
#
import numpy as np

class ZigZag():
    def __init__(self):
        self.idx = []
        
    def zig_zag(self, i, j, n):
        if i + j >= n:
            return n * n - 1 - self.zig_zag(n - 1 - i, n - 1 - j, n)
        k = (i + j) * (i + j + 1) // 2
        return k + i if (i + j) & 1 else k + j

    def zig_zag_getIdx(self, N):
        idx = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                idx[i, j] = self.zig_zag(i, j, N)
        return idx.reshape(-1)
    
    def transform(self, X):
        self.idx = self.zig_zag_getIdx((int)(np.sqrt(X.shape[-1]))).astype('int32')
        S = list(X.shape)
        X = X.reshape(-1, X.shape[-1])
        return X[:, np.argsort(self.idx)].reshape(S)
    
    def inverse_transform(self, X):
        self.idx = self.zig_zag_getIdx((int)(np.sqrt(X.shape[-1]))).astype('int32')
        S = list(X.shape)
        X = X.reshape(-1, X.shape[-1])
        return X[:, self.idx].reshape(S)