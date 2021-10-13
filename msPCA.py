# use cw Saab for PCA
# 2021.10.06
# @yifan
import numpy as np
from cwSaab import cwSaab
from util import Shrink, invShrink

def Concat(X, concatArg):
    return X

class msPCA():
    def __init__(self, n_components, win):
        self.n_components = n_components
        self.sSaabArg = {'num_AC_kernels':-1, 'needBias':False, 'useDC':False, 'batch':None}
        self.sshrinkArgs = {'func':Shrink, 'win':2}
        self.sinv_shrinkArgs = {'func':invShrink, 'win':2}
        self.concatArg = {'func':Concat}
        self.inv_concatArg = {'func':Concat}
        self.cwSaab = None
        self.depth = (int)(np.log2(win))
        self.win = win
        self.tComp = -1

    def duplicate(self, n, Arg):
        tmp = []
        for i in range(n):
            tmp.append(Arg)
        return tmp

    def fit(self, X):
        self.cwSaab = cwSaab(depth=self.depth, 
                             energyTH=0.,
                             SaabArgs=self.duplicate(self.depth, self.sSaabArg), 
                             shrinkArgs=self.duplicate(self.depth, self.sshrinkArgs), 
                             inv_shrinkArgs = self.duplicate(self.depth, self.sinv_shrinkArgs),
                             concatArg=self.concatArg,
                             inv_concatArg=self.inv_concatArg,
                             splitMode=2, 
                             cwHop1=True
                             ).fit(X)
        return self

    def trunc(self, tX):
        if self.n_components <= 0:
            return tX
        return tX[:,:,:,:self.n_components]

    def inv_trunc(self, tX):
        if self.n_components <= 0:
            return tX
        tmp = np.zeros((tX.shape[0], tX.shape[1], tX.shape[2], self.tComp))
        tmp[:,:,:,:self.n_components] = tX
        return tmp

    def transform(self, X):
        tX = self.cwSaab.transform(X)
        tX = tX[-1]
        self.tComp = tX.shape[-1]
        return self.trunc(tX)

    def inverse_transform(self, tX):
        tX = self.inv_trunc(tX)
        tmp = []
        for i in range(0, self.depth-1):
            tmp.append(np.zeros((tX.shape[0], 
                                 tX.shape[1]*pow(2, self.depth-1-i), 
                                 tX.shape[2]*pow(2, self.depth-1-i), 
                                 tX.shape[-1]//pow(4, self.depth-1-i))))
        tmp.append(tX)
        X = self.cwSaab.inverse_transform(tmp)
        return X
        
   
if __name__ == "__main__":
    from load_img import Load_from_Folder
    
    Y = Load_from_Folder('/Users/alex/Desktop/proj/compression/data/test_256/', 'RGB', ct=4)
    Y = np.array(Y)
    cw = msPCA(-1, 8)
    cw.fit(Y)
    tY = cw.transform(Y)
    print(tY.shape)
    iY = cw.inverse_transform(tY)
    print('PSNR:', 10*np.log10(255**2/np.mean(np.square(Y-iY))))