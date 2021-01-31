# 2021.01.27
# @yifan 
#
import numpy as np

from framework.core.dct import DCT
from framework.core.transform_utli import Shrink, invShrink

class BH_DCT():
    def __init__(self, depth=2, Win_list=[4, 4]):
        self.depth = depth
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