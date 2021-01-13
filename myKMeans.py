# @yifan
# 2021.01.12
#

import numpy as np 
from sklearn import cluster

class myKMeans():
    def __init__(self, n_clusters=-1, trunc=-1):
        self.KM          = cluster.KMeans(  n_clusters=n_clusters, n_init=11  )
        self.cent        = []
        self.trunc       = trunc

    def truncate(self, X):
        if self.trunc != -1:
            X[:, self.trunc:] *= 0
        return X
    
    def fit(self, X):
        X = X.reshape(  -1, X.shape[-1]  )
        self.truncate(X)
        self.KM.fit(  X  )
        self.cent = np.array(  self.KM.cluster_centers_  )
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
        X = self.cent[idx]
        return X.reshape(S)