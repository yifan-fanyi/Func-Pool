# 2020
# @yifan
#

import numpy as np 
from util import Shrink, invShrink
from myPCA import myPCA
from myKMeans import myKMeans

class PPVQ():
    def __init__(self, N_clusters, win, N_group=1):
        self.win = win
        self.N_clusters = N_clusters
        self.N_group = N_group
        self.PCA = myPCA()
        self.KM_list  = []

    def fit(self, X):
        X = Shrink(X, win=self.win)
        self.PCA.fit(X)
        X = self.PCA.transform(X)
        for i in range(0, X.shape[-1], self.N_group):
            km = myKMeans(self.N_clusters)
            km.fit(X[:,:,:,i:i+self.N_group])
            self.KM_list.append(km)
        return self
    
    def quantize(self, X):
        X = Shrink(X, win=self.win)
        X = self.PCA.transform(X)
        idx_list = []
        for i in range(len(self.KM_list)):
            t_idx = self.KM_list[i].predict(X[:,:,:,i:i+self.N_group])
            idx_list.append(t_idx)
        return idx_list
    
    def de_quantize(self, idx_list):
        X = []
        for i in range(len(self.KM_list)):
            t_X = self.KM_list[i].inv_predict(idx_list[i])
            X.append(t_X)
        X = np.concatenate(X, axis=-1)
        X = self.PCA.inverse_transform(X)
        X = invShrink(X, win=self.win)
        return X