# 2021.01.27
# @yifan & zhanxuan
# PCA transformation 
#
# 2D PCA modified from https://blog.csdn.net/w450468524/article/details/54895477
#
import numpy as np

class myPCA2D():
    def __init__(self, n_components, H=None, W=None):
        self.H            = H
        self.W            = W
        self.K1           = []
        self.K2           = []

    def fit(self, X):
        # input: sample: (X, Y, Z)
        #                 X is the index of different block
        #                 Y and Z is the height and width of block
        #        row_top and col_top are the height and width of block
        # output:
        #        X: the horizontal kernel
        #        Z: the vertical kernel
        # Notes:
        #        forward transform: C = Z.T*A*X
        #        inverse transform: A = Z*C*X.T
        S = X.shape
        X = X.reshape(-1, self.W, self.H)
        mean = np.zeros((self.W, self.H))
        for i in range(X.shape[0]):
            mean = mean + X[i]
        mean /= float(X.shape[0])
        cov_row = np.zeros((self.H, self.H))
        for i in range(X.shape[0]):
            diff = X[i] - mean
            cov_row = cov_row + np.dot(diff.T, diff)
        cov_row /= float(X.shape[0])
        row_eval, row_evec = np.linalg.eig(cov_row)
        sorted_index = np.argsort(row_eval)
        self.K1 = np.array(row_evec[:,sorted_index[:-self.H-1 : -1]]) 
        cov_col = np.zeros((self.W, self.W))
        for i in range(X.shape[0]):
            diff = X[i] - mean
            cov_col += np.dot(diff,diff.T)
        cov_col /= float(X.shape[0])
        col_eval, col_evec = np.linalg.eig(cov_col)
        sorted_index = np.argsort(col_eval)
        self.K2 = np.array(col_evec[:,sorted_index[:-self.W-1 : -1]])
        return self

    def trans(self, X, inv=False):
        res = []
        S = X.shape
        X = X.reshape(-1, self.W, self.H)
        for i in range(X.shape[0]):
            if inv == False:
                res.append(np.dot(np.transpose(self.K2), np.dot(X[i], self.K1)))
            else:
                res.append(np.dot(self.K2, np.dot(X[i], np.transpose(self.K1))))
        return np.array(res).reshape(S)

    def transform(self, X):
        return self.trans(X, inv=False)
        
    def inverse_transform(self, X):
        return self.trans(X, inv=True)
