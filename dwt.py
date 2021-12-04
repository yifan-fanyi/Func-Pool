import numpy as np

class MeanDiff_DWT():
    def __init__(self, level):
        self.level = level
        
    def transform_onelevel(self, X):
        tmp, res = np.zeros_like(X), np.zeros_like(X)
        hf = X.shape[0] // 2
        for i in range(0, X.shape[0], 2):
            pos = i // 2
            tmp[pos] = X[i] + X[i+1]
            tmp[pos+hf] = X[i] - X[i+1]
        for i in range(0, X.shape[1], 2):
            pos = i // 2
            res[:, pos] = tmp[:, i] + tmp[:, i+1]
            res[:, pos+hf] = tmp[:, i] - tmp[:, i+1]
        return res
        
    def inverse_transform_onelevel(self, res):
        tmp, X = np.zeros_like(res), np.zeros_like(res)
        hf = res.shape[0] // 2
        for i in range(0, res.shape[0], 2):
            pos = i // 2
            tmp[i] = (res[pos]+res[pos+hf])/2
            tmp[i+1] = (res[pos]-res[pos+hf])/2
        for i in range(0, X.shape[1], 2):
            pos = i // 2
            X[:, i] = (tmp[:, pos] + tmp[:, pos+hf])/2
            X[:, i+1] = (tmp[:, pos] - tmp[:, pos+hf])/2
        return X
    
    def W_gen(self, S):
        def setv(w):
            S = w.shape[0] // 2
            w[:S, :S] *=4
            w[S:, :S] *=2
            w[:S, S:] *=2
            return w
        w = np.ones((S, S))
        for i in range(self.level):
            w[:S,:S] = setv(w[:S,:S])
            S = S // 2
        return w
    
    def transform(self, X):
        assert X.shape[0] == X.shape[1]
        res = copy.deepcopy(X)
        S = X.shape[0]
        for i in range(self.level):
            res[:S, :S] = self.transform_onelevel(res[:S, :S])
            S = S // 2            
        return res / self.W_gen(X.shape[0])
    
    def inverse_transform(self, res):
        res *= self.W_gen(res.shape[0])
        S = res.shape[0] >> (self.level-1)
        X = copy.deepcopy(res)
        for i in range(self.level):
            X[:S, :S] = self.inverse_transform_onelevel(X[:S, :S])
            S = S * 2            
        return X.astype('int16')
    
if __name__ == "__main__":
    dwt = MeanDiff_DWT(1)
    a =np.array( [13, 26, 8, 29, ]).reshape(2,2)
    print(a)
    b = dwt.transform(a)
    print(b)
    print('---------------------------')
    c = dwt.inverse_transform(b)
    print(c)