# @yifan
# 2021.05.10
# add gpu and n_threads control for fast_KMeans
import numpy as np 
import sklearn
from sklearn import cluster
from framework.core.fast_kmeans import fast_KMeans

class myKMeans():
    def __init__(self, n_clusters=-1, trunc=-1, fast=True, gpu=False, n_threads=10):
        if fast == True:
            self.KM          = fast_KMeans(  n_clusters=n_clusters, n_init=11 , gpu=gpu, n_threads=n_threads)
            self.__version   = self.KM.__version__
        else:
            self.KM          = cluster.KMeans(  n_clusters=n_clusters, n_init=11  )
            self.__version__ =  sklearn. __version__ 
        self.n_clusters = n_clusters
        self.cluster_centers_        = []
        self.trunc       = trunc
        
    def truncate(self, X):
        if self.trunc != -1:
            X[:, self.trunc:] *= 0
        return X
    
    def fit(self, X):
        X = X.reshape(  -1, X.shape[-1]  )
        self.truncate(X)
        self.KM.fit(  X  )
        self.cluster_centers_ = np.array(  self.KM.cluster_centers_  )
        return self

    def predict(self, X):
        S = (list)(X.shape)
        S[-1] = -1
        X = X.reshape(-1, X.shape[-1])
        idx = self.KM.predict(X)
        return idx.reshape(S)

    def inverse_predict(self, idx):
        S = (list)(idx.shape)
        S[-1] = -1
        idx = idx.reshape(-1,)
        X = self.cluster_centers_[idx]
        return X.reshape(S)



        