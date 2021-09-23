# 2021.01.27
# @yifan 
#
import numpy as np
import copy

from myPCA import myPCA
from util import Shrink, invShrink

class BH_PCA():
    def __init__(self, depth):
        self.depth = depth
        self.PCA_list = []
        self.Win_list = []
        
    def fit(self, X, win=4, is2D=True):
        for i in range(self.depth):
            tPCA, tXtmp = [], []
            for k in range(X.shape[-1]):
                tmp = Shrink(copy.deepcopy(X[:,:,:,k:k+1]), win=win)
                pca = myPCA(is2D=is2D, H=win, W=win).fit(tmp)
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
    
    def fit_single_hop(self, pX, win=4, is2D=True):
        tPCA, tXtmp = [], []
        for k in range(pX.shape[-1]):
            tmp = Shrink(copy.deepcopy(pX[:,:,:,k:k+1]), win=win)
            pca = myPCA(is2D=True, H=win, W=win).fit(tmp)
            tPCA.append(pca)
        self.PCA_list.append(tPCA)
        self.Win_list.append(win)
        return self
            
    def transform_single_hop(self, pX, hop):
        tXtmp = []
        for k in range(pX.shape[-1]):
            tmp = Shrink(copy.deepcopy(pX[:,:,:,k:k+1]), win=self.Win_list[hop-1])
            tmp = self.PCA_list[hop-1][k].transform(tmp)
            tXtmp.append(tmp)
        return np.concatenate(tXtmp, axis=-1)
    
    def inverse_transform_single_hop(self, pX, hop):
        iXtmp = []
        for k in range(0, pX.shape[-1], self.Win_list[hop-1]**2):
            iX = self.PCA_list[hop-1][k//self.Win_list[hop-1]**2].inverse_transform(pX[:,:,:,k:k+self.Win_list[hop-1]**2])
            iX = invShrink(iX, win=self.Win_list[hop-1])
            iXtmp.append(iX)
        return np.concatenate(iXtmp, axis=-1)
