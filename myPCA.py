# v2020.05.22
import numpy as np 
from sklearn.decomposition import PCA

class myPCA():
    def __init__(self, isInteger=True, bits=12, opType='int64'):
        self.trained = False
        self.Kernels = []
        self.PCA = None
        self.Energy = []
        self.bits = bits
        self.isInteger = isInteger
        self.opType = opType

    def to_int_(self):
        self.Kernels = np.round(self.Kernels * pow(2, self.bits)).astype(self.opType)

    def fit(self, X):
        S = X.shape
        self.PCA = PCA(n_components=X.shape[-1])
        self.PCA.fit(X.reshape(-1, X.shape[-1]))
        self.Kernels = self.PCA.components_
        self.Energy = self.PCA.explained_variance_ratio_
        self.trained = True
        if self.isInteger == True:
            self.to_int_()
        return self

    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        return np.dot(X, np.transpose(self.Kernels))

    def inverse_transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        X = np.dot(X, self.Kernels)
        if self.isInteger == True:
            X = np.round(X / pow(2, 2 * self.bits)).astype(self.opType)
        return X

if __name__ == "__main__":
    import cv2
    X = cv2.imread('/Users/alex/Desktop/proj/compression/data/kodim03.png')
    p = myPCA()
    p.fit(X)
    Y = p.transform(X)
    Y = p.inverse_transform(Y)
    print(np.mean(np.abs(Y-X)))

