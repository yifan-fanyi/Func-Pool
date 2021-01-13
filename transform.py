# @yifan
# v2020.01.12
# transformation 
#

import numpy as np 
import copy
from scipy.fftpack import dct, idct
from utli import Shrink, invShrink, myPCA

class BH_PCA():
    def __init__(self, depth):
        self.depth = depth
        self.PCA_list = []
        self.Win_list = []
        
    def fit(self, X, win=4):
        for i in range(self.depth):
            tPCA, tXtmp = [], []
            for k in range(X.shape[-1]):
                tmp = Shrink(copy.deepcopy(X[:,:,:,k:k+1]), win=win)
                pca = myPCA().fit(tmp)
                tPCA.append(pca)
                tmp = pca.transform(tmp)
                tXtmp.append(tmp)
            self.PCA_list.append(tPCA)
            self.Win_list.append(win)
            X = np.concatenate(tXtmp, axis=-1)
        return self
    
    def transform(self, X):
        tX = []
        for i in range(0, self.depth):
            tXtmp = []
            for k in range(X.shape[-1]):
                tmp = Shrink(copy.deepcopy(X[:,:,:,k:k+1]), win=self.Win_list[i])
                tmp = self.PCA_list[i][k].transform(tmp)
                tXtmp.append(tmp)
            X = np.concatenate(tXtmp, axis=-1)
            tX.append(copy.deepcopy(X))
        return tX
                
    def inverse_transform(self, tX):
        for i in range(len(tX)-1, 0, -1):
            for k in range(0, tX[i].shape[-1], self.Win_list[i]**2):
                iX = self.PCA_list[i][k//self.Win_list[i]**2].inverse_transform(tX[i][:,:,:,k:k+self.Win_list[i]**2])
                tX[i-1][:,:,:,k//self.Win_list[i]**2:k//self.Win_list[i]**2+1] = invShrink(iX, win=self.Win_list[i])
        iX = self.PCA_list[0][0].inverse_transform(tX[0])
        iX = invShrink(iX, win=self.Win_list[0])
        return iX
    
    def fit_single_hop(self, pX, win=4):
        tPCA, tXtmp = [], []
        for k in range(pX.shape[-1]):
            tmp = Shrink(copy.deepcopy(pX[:,:,:,k:k+1]), win=win)
            pca = myPCA().fit(tmp)
            tPCA.append(pca)
        self.PCA_list.append(tPCA)
        self.Win_list.append(win)
            
    def transform_single_hop(self, pX, hop):
        tXtmp = []
        for k in range(pX.shape[-1]):
            tmp = Shrink(copy.deepcopy(pX[:,:,:,k:k+1]), win=self.Win_list[hop])
            tmp = self.PCA_list[hop][k].transform(tmp)
            tXtmp.append(tmp)
        return np.concatenate(tXtmp, axis=-1)
    
    def inverse_transform_single_hop(self, pX, hop):
        iXtmp = []
        for k in range(0, pX.shape[-1], self.Win_list[hop]**2):
            iX = self.PCA_list[hop][k//self.Win_list[hop]**2].inverse_transform(pX[:,:,:,k:k+self.Win_list[hop]**2])
            iX = invShrink(iX, win=self.Win_list[hop])
            iXtmp.append(iX)
        return np.concatenate(iXtmp, axis=-1)

class DCT():
    def __init__(self, N=8, P=8):
        self.N = N
        self.P = P
    
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

class BH_DCT():
    def __init__(self, depth, Win_list=[]):
        self.depth = depth
        self.PCA_list = []
        self.Win_list = Win_list
    
    def transform(self, X):
        tX = []
        for i in range(0, self.depth):
            tXtmp = []
            for k in range(X.shape[-1]):
                tmp = Shrink(copy.deepcopy(X[:,:,:,k:k+1]), win=self.Win_list[i])
                tmp = DCT(self.Win_list[i], self.Win_list[i]).transform(tmp)
                tXtmp.append(tmp)
            X = np.concatenate(tXtmp, axis=-1)
            tX.append(copy.deepcopy(X))
        return tX
                
    def inverse_transform(self, tX):
        for i in range(len(tX)-1, 0, -1):
            for k in range(0, tX[i].shape[-1], self.Win_list[i]**2):
                iX = DCT(self.Win_list[i], self.Win_list[i]).inverse_transform(tX[i][:,:,:,k:k+self.Win_list[i]**2])
                tX[i-1][:,:,:,k//self.Win_list[i]**2:k//self.Win_list[i]**2+1] = invShrink(iX, win=self.Win_list[i])
        iX = DCT(self.Win_list[i], self.Win_list[i]).inverse_transform(tX[0])
        iX = invShrink(iX, win=self.Win_list[0])
        return iX
            
    def transform_single_hop(self, pX, hop):
        tXtmp = []
        for k in range(pX.shape[-1]):
            tmp = Shrink(copy.deepcopy(pX[:,:,:,k:k+1]), win=self.Win_list[hop])
            tmp = DCT(self.Win_list[hop], self.Win_list[hop]).transform(tmp)
            tXtmp.append(tmp)
        return np.concatenate(tXtmp, axis=-1)
    
    def inverse_transform_single_hop(self, pX, hop):
        iXtmp = []
        for k in range(0, pX.shape[-1], self.Win_list[hop]**2):
            iX = DCT(self.Win_list[hop], self.Win_list[hop]).inverse_transform(pX[:,:,:,k:k+self.Win_list[hop]**2])
            iX = invShrink(iX, win=self.Win_list[hop])
            iXtmp.append(iX)
        return np.concatenate(iXtmp, axis=-1)