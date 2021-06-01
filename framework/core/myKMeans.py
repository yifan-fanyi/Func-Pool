# @yifan
# add gpu and n_threads control for fast_KMeans
import numpy as np 
import copy

from numpy.lib.npyio import save
import sklearn
from sklearn import cluster
from sklearn.metrics.pairwise import euclidean_distances
from framework.core.fast_kmeans import fast_KMeans

def sort_by_eng(Cent):
    eng = np.sum(np.square(Cent), axis=1)
    idx = np.argsort(eng)
    mp, imp = {}, {}
    for i in range(len(idx)):
        assert (i not in mp.keys()), "Err"
        assert (idx[i] not in imp.keys()), 'err'
        mp[i] = idx[i]
        imp[idx[i]] = i
    return mp, imp

class Mapping():
    def __init__(self, Cent):
        self.map, self.inv_map = sort_by_eng(Cent)
        self.version = '2021.05.14'

    def transform(self, label):
        S = label.shape
        label = label.reshape(-1)
        for i in range(len(label)):
            label[i] = self.map[label[i]]
        return label.reshape(S)
    
    def inverse_transform(self, l):
        S = l.shape
        label = copy.deepcopy(l).reshape(-1)
        for i in range(len(label)):
            label[i] = self.inv_map[label[i]]
        return label.reshape(S)

class myKMeans():
    def __init__(self, n_clusters=-1, trunc=-1, fast=True, gpu=False, n_threads=10, sort=False, saveObj=False):
        if fast == True:
            self.KM          = fast_KMeans(  n_clusters=n_clusters, n_init=11 , gpu=gpu, n_threads=n_threads)
            self.version_   = self.KM.__version__
        else:
            self.KM          = cluster.KMeans(  n_clusters=n_clusters, n_init=11  )
            self.version_ =  sklearn. __version__ 
        self.n_clusters = n_clusters
        self.cluster_centers_        = []
        self.trunc       = trunc
        self.sort        = sort
        self.saveObj     = saveObj
        self.version = '2021.05.31'
        
    def truncate(self, X):
        if self.trunc != -1:
            X[:, self.trunc:] *= 0
        return X
    
    def fit(self, X):
        X = X.reshape(  -1, X.shape[-1]  )
        self.truncate(X)
        self.KM.fit(  X  )
        if self.sort == True:
            self.MP = Mapping(np.array(  self.KM.cluster_centers_  ))
        self.cluster_centers_ = np.array(  self.KM.cluster_centers_  )
        if self.saveObj == False:
            self.KM = None
        return self

    def Cpredict(self, X):
        dis = euclidean_distances(X, self.cluster_centers_)
        pred = np.argmin(dis, axis=1)
        return pred

    def predict(self, X):
        S = (list)(X.shape)
        S[-1] = -1
        X = X.reshape(-1, X.shape[-1])
        if self.saveObj == True:
            idx = self.KM.predict(X)
        else:
            idx = self.Cpredict(X)
        if self.sort == True:
            idx = self.MP.transform(idx)
        return idx.reshape(S)

    def inverse_predict(self, idx):
        S = (list)(idx.shape)
        S[-1] = -1
        idx = idx.reshape(-1,)
        if self.sort == True:
            idx = self.MP.inverse_transform(idx)
        X = self.cluster_centers_[idx]
        return X.reshape(S)