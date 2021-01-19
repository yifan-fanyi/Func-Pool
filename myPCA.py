# v2021.01.19
import numpy as np 
from sklearn.decomposition import PCA

class myPCA():
    def __init__(self, n_components=-1, is2D=False, H=None, W=None):
        self.is2D         = is2D
        if is2D == False:
            self.n_components = n_components
            self.Kernels      = []
            self.PCA          = None
            self.Energy_ratio = []
            self.Energy       = []
        else:     
            self.H            = H
            self.W            = W
            self.K1           = []
            self.K2           = []
    
    def PCA_2D_fit(self, X):
        S = X.shape
        X = X.reshape(-1, self.W, self.H)
        mean = np.zeros((self.W, self.H))
        for i in range(X.shape[0]):
            mean = mean + X[i]
        mean /= float(X.shape[0])
        cov_row = np.zeros((self.H, self.H))
        for i in range(X.shape[0]):
            diff = X[i] - mean
            cov_row = cov_row + np.dot(diff.T, diff)
        cov_row /= float(X.shape[0])
        row_eval, row_evec = np.linalg.eig(cov_row)
        sorted_index = np.argsort(row_eval)
        self.K1 = row_evec[:,sorted_index[:-self.H-1 : -1]]
        # m*m matrix
        cov_col = np.zeros((self.W, self.W))
        for i in range(X.shape[0]):
            diff = X[i] - mean
            cov_col += np.dot(diff,diff.T)
        cov_col /= float(X.shape[0])
        col_eval, col_evec = np.linalg.eig(cov_col)
        sorted_index = np.argsort(col_eval)
        self.K2 = col_evec[:,sorted_index[:-self.W-1 : -1]]
        
    def PCA_2D_transform(self, X, inv=False):
        res = []
        S = X.shape
        X = X.reshape(-1, self.W, self.H)
        for i in range(X.shape[0]):
            if inv == False:
                res.append(np.dot(self.K2.T, np.dot(X[i], self.K1)))
            else:
                res.append(np.dot(self.K2, np.dot(X[i], self.K1.T)))
        res = np.concatenate(res, axis=0)
        return res.reshape(S)
    
    def PCA_sklearn(self, X):
        self.PCA          = PCA(  n_components=self.n_components  )
        self.PCA.fit(X)
        self.Kernels      = self.PCA.components_
        self.Energy_ratio = self.PCA.explained_variance_ratio_
        self.Energy       = self.PCA.explained_variance_
        
    def PCA_numpy(self, X):
        X = X - np.mean(X.copy(), axis=0)
        X_cov = np.cov(X, rowvar=0)
        eVal, eVect = np.linalg.eig(X_cov)
        idx = np.argsort(eVal)[::-1]
        idx = idx[:self.n_components]
        self.Kernels = np.transpose(eVect[:, idx])
        self.Energy_ratio = eVal / np.sum(eVal)
        self.Energy_ratio = self.Energy_ratio[idx]
        self.Energy = eVal[idx]
        
    def fit(self, X, whichPCA='sklearn'):
        if self.is2D == True:
            self.PCA_2D_fit(X)
        else:
            X = X.reshape(  -1, X.shape[-1]  )
            if self.n_components == -1:
                self.n_components = X.shape[-1]
            if whichPCA == 'sklearn':
                self.PCA_sklearn(  X  )
            elif whichPCA == 'numpy':
                self.PCA_numpy(X)
            else:
                assert (False), "whichPCA only support 'numpy' or 'sklearn' when is2D==False!"
        return self
            
    def transform(self, X):
        if self.is2D == False:
            return np.dot(  X, np.transpose(  self.Kernels  )  )
        else:
            return self.PCA_2D_transform(X, inv=False)

    def inverse_transform(self, X):
        if self.is2D == False:
            return np.dot(  X, self.Kernels  )
        else:
            return self.PCA_2D_transform(X, inv=True)

if __name__ == "__main__":
    import cv2
    X = cv2.imread('/Users/alex/Desktop/proj/compression/data/Kodak/kodim03.png')
    p = myPCA(n_components=2, isInteger=1)
    p.fit(X)
    print(p.Kernels)
    print(p.Energy_ratio)
    Y = p.transform(X)
    Y = p.inverse_transform(Y)
    print(np.mean(np.abs(Y-X)))
    print()

    p.fit(X, whichPCA='sklearn')
    print(p.Kernels)
    print(p.Energy_ratio)

    Y = p.transform(X)
    Y = p.inverse_transform(Y)
    print(np.mean(np.abs(Y-X)))

