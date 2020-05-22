# v2020.05.22
import numpy as np 
from sklearn.decomposition import PCA

class myPCA():
    def __init__(self):
        self.trained = False
        self.Kerenls = []
        self.PCA = None
        self.Energy = []

    def fit(self, X):
        S = X.shape
        self.PCA = PCA(n_components=X.shape[-1])
        self.PCA.fit(X.reshape(-1, X.shape[-1]))
        self.Kerenls = self.PCA.components_
        self.Energy = self.PCA.explained_variance_ratio_
        self.trained = True
        return self

    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        return np.dot(X, np.transpose(self.Kerenls))

    def inverse_transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        return np.dot(X, self.Kerenls)

if __name__ == "__main__":
    import cv2
    X = cv2.imread('/Users/alex/Desktop/proj/compression/data/kodim03.png')
    p = myPCA()
    p.fit(X)
    Y = p.transform(X)
    Y = p.inverse_transform(Y)
    print(np.mean(np.abs(Y-X)))

