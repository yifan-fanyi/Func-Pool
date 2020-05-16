# v2020.05.14
# block hierarchy
import numpy as np 

from framework.cwSaab import cwSaab
import warnings
warnings.filterwarnings("ignore")

class BH(cwSaab):
    def __init__(self, block=4, depth=3, TH1=0.001, TH2=0.0001, SaabArgs=None, shrinkArgs=None, inv_shrinkArgs=None):
        super().__init__(depth=depth, 
                         energyTH=TH1, 
                         SaabArgs=SaabArgs,
                         shrinkArgs=shrinkArgs, 
                         concatArg={'func':self.concat}, 
                         splitMode=0, 
                         cwHop1=True)
        self.block = block
        self.TH2 = TH2
        self.outShape = []
        self.inv_shrinkArgs = inv_shrinkArgs
        self.saveidx = []
         
    def concat(self, X, Arg):
        return X

    def fit(self, X):
        assert ((X.shape[1] % pow(self.block, self.depth)) == 0), "Input shape must be exactly match the block size, axis 1 not match!"
        assert ((X.shape[2] % pow(self.block, self.depth)) == 0), "Input shape must be exactly match the block size, axis 2 not match!" 
        super().fit(X)
        for i in range(self.depth-1):
            idx = []
            for j in range(len(self.splitidx[i])):
                idx.append(self.splitidx[i][j] == False and self.Energy[i][j] >= self.TH2)
            self.saveidx.append(idx)
        idx = []
        for j in range(len(self.Energy[-1])):
            idx.append(self.Energy[-1][j] >= self.TH2)
        self.saveidx.append(idx)
        return self

    def encode(self, X):
        print(len(self.saveidx))
        X = self.transform(X)
        for i in range(self.depth-1):
            self.outShape.append(X[i].shape)
            X[i] = X[i][:, :, :, self.saveidx[i]]
        return X 

    def decode(self, X):
        tmp = []
        for i in range(self.depth-1):
            tt = np.zeros(self.outShape[i])
            ct = 0
            for j in range(len(self.saveidx[i])):
                if self.saveidx[i][j] == True:
                    tt[:, :, :, j] = X[i][:, :, :, ct]
                    ct += 1
            tmp.append(tt)
        tmp.append(X[-1])
        X = self.inverse_transform(tmp, inv_concatArg={'func':self.concat}, inv_shrinkArgs=self.inv_shrinkArgs)
        return X
    
