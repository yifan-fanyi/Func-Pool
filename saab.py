# v2020.02.19

# Saab transformation
# modeiled from https://github.com/davidsonic/Interpretable_CNN

import numpy as np
from sklearn.decomposition import PCA
import time

class Saab():
    def __init__(self, num_kernels=-1, useDC=True, needBias=True):
        self.par = None
        self.Kernels = []
        self.Bias = []
        self.Mean0 = []
        self.Energy = []
        self.num_kernels = num_kernels
        self.useDC = useDC
        self.needBias = needBias
        self.trained = False

    def remove_mean(self, X, axis):
        feature_mean = np.mean(X, axis=axis, keepdims=True)
        X = X - feature_mean
        return X, feature_mean

    def fit(self, X): 
        assert (len(X.shape) == 2), "Input must be a 2D array!"
        X = X.astype('float32')
        X, self.Mean0 = self.remove_mean(X.copy(), axis=0)
        X, dc = self.remove_mean(X.copy(), axis=1)
        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]
        pca = PCA(n_components=self.num_kernels, svd_solver='full').fit(X)
        kernels = pca.components_
        energy = pca.explained_variance_ / np.sum(pca.explained_variance_)
        if self.useDC == True:  
            largest_ev = np.var(dc * np.sqrt(X.shape[-1]))     
            dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1])) / np.sqrt(largest_ev)
            kernels = np.concatenate((dc_kernel, kernels[:-1]), axis=0)
            energy = np.concatenate((np.array([largest_ev]), pca.explained_variance_[:-1]), axis=0)
            energy = energy / np.sum(energy)
        bias = np.linalg.norm(X, axis=1)
        bias = np.max(bias)
        self.Kernels, self.Energy, self.Bias = kernels, energy, bias
        self.trained = True
        
    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        X = X.astype('float32')
        X -= self.Mean0
        if self.needBias == True:
            X += self.Bias
        transformed = np.matmul(X, np.transpose(self.Kernels))
        if self.needBias == True:
            e = np.zeros((1, self.Kernels.shape[0]))
            e[0, 0] = 1
            transformed -= self.Bias*e
        return transformed



