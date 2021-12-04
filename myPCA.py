# 2021.01.27
# @yifan
# PCA transformation 
#
# 2D PCA modified from https://blog.csdn.net/w450468524/article/details/54895477
#
import numpy as np
from sklearn.decomposition import PCA

class myPCA():
    def __init__(self, n_components=-1):
        self.n_components = n_components
        self.Kernels      = []
        self.PCA          = None
        self.Energy_ratio = []
        self.Energy       = []
        self.__version__  = '2021.11.05' 

    def PCA_sklearn(self, X):
        self.PCA          = PCA(  n_components=self.n_components  )
        self.PCA.fit(X)
        self.Kernels      = self.PCA.components_
        self.Energy_ratio = self.PCA.explained_variance_ratio_
        self.Energy       = self.PCA.explained_variance_
        
    def fit(self, X):
        X = X.reshape(  -1, X.shape[-1]  )
        if self.n_components == -1:
            self.n_components = X.shape[-1]
        self.PCA_sklearn(  X  )  
        return self
            
    def transform(self, X):
        S = (list)(X.shape)
        S[-1] = -1
        X = X.reshape(  -1, X.shape[-1]  )
        tX = self.PCA.transform(X)
        return tX.reshape(S)

    def inverse_transform(self, X):
        S = (list)(X.shape)
        S[-1] = -1
        X = X.reshape(  -1, X.shape[-1]  )
        tX = self.PCA.inverse_transform(X)
        tX = tX.reshape(S)
        return tX
       