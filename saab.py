# v2020.05.29

# Saab transformation
# modeiled from https://github.com/davidsonic/Interpretable_CNN

import numpy as np
from sklearn.decomposition import PCA

from myPCA import myPCA
class Saab():
    def __init__(self, num_kernels=-1, useDC=True, needBias=True, isInteger=False, bits=8, opType='int64'):
        self.par = None
        self.Kernels = []
        self.Bias = []
        self.Mean0 = []
        self.Energy = []
        self.num_kernels = num_kernels
        self.useDC = useDC
        self.needBias = needBias
        self.trained = False
        self.bits = bits
        self.isInteger = isInteger
        self.opType = opType

    def remove_mean(self, X, axis):
        feature_mean = np.mean(X, axis=axis, keepdims=True)
        X = X - feature_mean
        return X, feature_mean
    
    def to_int_(self):
        assert (self.useDC == False), "Integer transformation is only supported when 'useDC=False'!"
        self.Bias = np.round(self.Bias * pow(2, self.bits)+1).astype(self.opType)
        self.Kernels = np.round(self.Kernels * pow(2, self.bits)).astype(self.opType)
        self.Mean0 = np.round(self.Mean0 * pow(2, self.bits)).astype(self.opType)
        
    def fit(self, X, whichPCA='numpy'): 
        assert (len(X.shape) == 2), "Input must be a 2D array!"
        X = X.astype('float32')
        if self.useDC == True:
            X, dc = self.remove_mean(X.copy(), axis=1)
        X, self.Mean0 = self.remove_mean(X.copy(), axis=0)
        self.Bias = np.max(np.linalg.norm(X, axis=1)) * 1 / np.sqrt(X.shape[1])
        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]
        pca = myPCA(n_components=self.num_kernels, isInteger=False)
        pca.fit(X, whichPCA=whichPCA)
        kernels = pca.Kernels
        energy = pca.Energy_ratio
        if self.useDC == True:  
            largest_ev = np.var(dc * np.sqrt(X.shape[-1]))     
            dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1])) / np.sqrt(largest_ev)
            kernels = np.concatenate((dc_kernel, kernels[:-1]), axis=0)
            energy = np.concatenate((np.array([largest_ev]), pca.Energy[:-1]), axis=0)
            energy = energy / np.sum(energy)
        self.Kernels, self.Energy = kernels, energy
        if self.isInteger == True:
            self.to_int_()
        self.trained = True
        return self
        
    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        X = X.astype('float32')
        if self.useDC == True:
            X -= self.Mean0
        if self.needBias == True:
            X += self.Bias
        X = np.matmul(X, np.transpose(self.Kernels))
        if self.needBias == True and self.useDC == True:
            X[:, 0] -= self.Bias
        if self.isInteger == True:
            X = X.astype(self.opType)
        return X
    
    def inverse_transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        if (self.useDC == True):
            print("       <Warning> May result larger reconstruction error!")
        X  = X.astype('float32')
        if self.needBias == True and self.useDC == True:
            X[:, 0] += self.Bias
        X = np.matmul(X, self.Kernels)
        if self.needBias == True:
            X -= self.Bias 
        if self.useDC == True:
            X += self.Mean0
        if self.isInteger == True:
            X = np.round(X / pow(2, 2 * self.bits)).astype(self.opType)
        return X

if __name__ == "__main__":
    from sklearn import datasets
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    data = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(data.shape))
    print(" --> test inv")
    print(" -----> num_kernels=-1, needBias=False, useDC=True")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=True, needBias=False)
    saab.fit(X, whichPCA='numpy')
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    print(np.mean(np.abs(X-Y)))
    #assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, needBias=True, useDC=True")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=True, needBias=True)
    saab.fit(X)
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    print(np.mean(np.abs(X-Y)))
    #assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, needBias=False, useDC=False")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=False, needBias=False)
    saab.fit(X)
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, needBias=True, useDC=False")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=False, needBias=True)
    saab.fit(X)
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"
    print(" -----> num_kernels=-1, needBias=True, useDC=False, isInteger=True")
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    saab = Saab(num_kernels=-1, useDC=False, needBias=False, isInteger=True, bits=16)
    saab.fit(X)
    Xt = saab.transform(X)
    Y = saab.inverse_transform(Xt)
    assert (np.mean(np.abs(X-Y)) < 1e-5), "invSaab error!"

