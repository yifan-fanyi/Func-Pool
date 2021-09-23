# 2021.01.27
# @yifan 
#
import numpy as np
import copy

from dct import DCT
from util import Shrink, invShrink

class BH_DCT():
    def __init__(self, depth=5):
        self.depth = depth
        self.Win_list = []
        
    def fit(self, X, win=4, is2D=True):
        return self

    def fit_single_hop(self, pX, win=4, is2D=True):
        return self

    def transform(self, X, win=4):
        tX = []
        for i in range(0, self.depth):
            tXtmp = []
            self.Win_list.append(win)
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
            
    def transform_single_hop(self, pX, hop, win=4):
        tXtmp = []
        self.Win_list.append(win)
        for k in range(pX.shape[-1]):
            tmp = Shrink(copy.deepcopy(pX[:,:,:,k:k+1]), win=self.Win_list[hop-1])
            tmp = DCT(self.Win_list[hop-1], self.Win_list[hop-1]).transform(tmp)
            tXtmp.append(tmp)
        return np.concatenate(tXtmp, axis=-1)
    
    def inverse_transform_single_hop(self, pX, hop):
        iXtmp = []
        for k in range(0, pX.shape[-1], self.Win_list[hop-1]**2):
            iX = DCT(self.Win_list[hop-1], self.Win_list[hop-1]).inverse_transform(pX[:,:,:,k:k+self.Win_list[hop-1]**2])
            iX = invShrink(iX, win=self.Win_list[hop-1])
            iXtmp.append(iX)
        return np.concatenate(iXtmp, axis=-1)