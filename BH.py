# v2020.05.22
# block hierarchy
import numpy as np 

from framework.cwSaab import cwSaab

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
        self.trained = False
         
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
        self.trained = True
        return self

    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        X = super().transform(X)
        for i in range(self.depth-1):
            self.outShape.append(X[i].shape)
            X[i] = X[i][:, :, :, self.saveidx[i]]
        return X 

    def inverse_transform(self, X):
        assert (self.trained == True), "Must call fit first!"
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
        X = super().inverse_transform(tmp, inv_concatArg={'func':self.concat}, inv_shrinkArgs=self.inv_shrinkArgs)
        return X
    
if __name__ == "__main__":
    from framework.evaluate import *
    from utli import *
    import cv2

    X = cv2.imread('Y.jpg', 0)
    print('input shape: ',X.shape)
    X = X.reshape(1, 512, 768, 1)

    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':False, 'batch':None, 'isInteger':True, 'bits':12, 'opType':'int64'}, 
                {'num_AC_kernels':-1, 'needBias':False, 'useDC':False, 'batch':None, 'isInteger':True, 'bits':12, 'opType':'int64'}]
    shrinkArgs = [{'func':Shrink, 'win':4}, 
                {'func': Shrink, 'win':4}]
    inv_shrinkArgs = [{'func':invShrink, 'win':4}, 
                    {'func': invShrink, 'win':4}]

    b = BH(block=4, depth=2, TH1=0.01, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, inv_shrinkArgs=inv_shrinkArgs)
    b.fit(X)
    eX = b.transform(X)
    dX = b.inverse_transform(eX)
    print('PSNR: ', PSNR(X, dX))