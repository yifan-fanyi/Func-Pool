# 2020.10.19
# @yifan
# channel-wise VQ
# input is asumed to be DCT/PCA coefficients
#
import numpy as np
from sklearn import cluster
from skimage.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances

from cwVQ_utli import *

def check_mse(X, km, PSNR_TH):
    mse_TH = 255**2 / pow(10, PSNR_TH / 10)
    idx = km.predict(X)
    res = km.cluster_centers_[idx]
    mse = mean_squared_error(X, res)
    if mse > mse_TH:
        return mse, False
    return mse, True
    
class cwVQ():
    # cw_idx: splitting point
    # cw_N: num codeword for each cluster
    def __init__(self, cw_idx, cw_N, PSNR_TH):
        self.cw_idx = cw_idx
        self.cw_N = cw_N
        self.PSNR_TH = PSNR_TH
        self.km_list = []
        self.cent_list = []
        self.dim = 0
        self.trained = False

    def fit(self, X):
        self.dim = X.shape[-1]
        print("       \033[32m---> cwVQ, num of raining smaples: %d"%(X.shape[0]))
        for i in range(1, len(self.cw_idx)):
            tmp = X[:, self.cw_idx[i-1]:self.cw_idx[i]]
            N = self.cw_N[i-1]
            while N < 200 * self.cw_N[i-1]:
                km = cluster.KMeans(n_clusters=int(N), n_init=7)
                print(np.std(tmp))
                km.fit(tmp)
                mse, flag = check_mse(tmp, km, self.PSNR_TH)
                #flag = True
                if flag == True:
                    print("          ---> MSE=%3f nice, stop"%(mse))
                    break
                N += 1
                print("          ---> MSE=%3f too large, increase N to %2d"%(mse, N))
            self.cw_N[i-1] = N
            print("     <INFO> Finish training feature idx %d - %d, with N=%d" %(self.cw_idx[i-1], self.cw_idx[i], self.cw_N[i-1]))
            km.cluster_centers_.sort(axis=0)
            self.km_list.append(km)
            self.cent_list.append(km.cluster_centers_)
        print("\033[0m")
        self.trained = True
    
    def encode(self, X):
        assert (self.trained == True), "   \033[0;91m<ERROR> Call fit first!\033[0m"
        idx = []
        for i in range(1, len(self.cw_idx)):
            tmp = X[:, self.cw_idx[i-1]:self.cw_idx[i]]
            tmp_idx = np.argmin(euclidean_distances(tmp, self.cent_list[i-1]), axis=1)#self.km_list[i-1].predict(tmp)
            idx.append(tmp_idx)
        return idx
    
    def decode(self, idx):
        assert (self.trained == True), "   \033[0;91m<ERROR> Call fit first!\033[0m"
        res = []
        print(idx[1][:10], self.cent_list[1][idx[1][:10]])
        for i in range(len(idx)):
            tmp = self.cent_list[i][idx[i]]
            res.append(tmp)
        res = np.concatenate(res, axis=1)
        if res.shape[-1] < self.dim:
            res = np.concatenate((res, np.zeros((res.shape[0], self.dim-res.shape[-1]))), axis=1)
        return res

class cwVQ4D(cwVQ):
    def __init__(self, cw_idx, cw_N, PSNR_TH, win, mode=0):
        super().__init__(cw_idx, cw_N, PSNR_TH)
        self.win = win
        self.mode = mode
        self.pca = myPCA(n_components=-1)

    def to2D(self, X, train=True):
        X = Shrink(X, {'win':self.win})
        if self.mode == 1:
            X = DCT(X)
            X = ZigZag().transform(X)
        elif self.mode == 2:
            if train == True:
                self.pca.fit(X)
            self.pca.transform(X)
        return X.reshape(-1, self.win**2), X.shape

    def to4D(self, X, S):
        X = X.reshape(S)
        if self.mode == 1:
            X = ZigZag().inverse_transform(X)
            X = IDCT(X)
        elif self.mode == 2:
            self.pca.inverse_transform(X)
        return invShrink(X, {'win':self.win})

    def fit(self, X):
        X, _ = self.to2D(X, train=True)
        super().fit(X)

    def encode(self, X):
        X, S = self.to2D(X, train=False)
        return super().encode(X), S
    
    def decode(self, idx, S):
        res = super().decode(idx)
        return self.to4D(res, S)

class kmVQ():
    def __init__(self, N):
        self.km = cluster.KMeans(n_clusters=int(N), n_init=7)
        self.cent = []
    def fit(self, X):
        print("       \033[32m---> VQ, num of raining smaples: %d"%(X.shape[0]))
        self.km.fit(X)
        self.cent = self.km.cluster_centers_
    def encode(self, X):
        return self.km.predict(X)
    def decode(self, idx):
        return self.cent[idx]

class kmVQ4D(kmVQ):
    def __init__(self, N, win, mode=0):
        super().__init__(N)
        self.win = win
        self.mode = mode
        self.mode = mode
        self.pca = myPCA(n_components=32)
    def to2D(self, X, train=True):
        X = Shrink(X, {'win':self.win})
        if self.mode == 1:
            X = DCT(X)
            X = ZigZag().transform(X)
        elif self.mode == 2:
            if train == True:
                self.pca.fit(X)
            self.pca.transform(X)
        return X.reshape(-1, self.win**2), X.shape
    def to4D(self, X, S):
        X = X.reshape(S)
        if self.mode == 1:
            X = ZigZag().inverse_transform(X)
            X = IDCT(X)
        elif self.mode == 2:
            self.pca.inverse_transform(X)
        return invShrink(X, {'win':self.win})
    def fit(self, X):
        X, _ = self.to2D(X, train=True)
        super().fit(X)
    def encode(self, X):
        X, S = self.to2D(X, train=False)
        return super().encode(X), S
    def decode(self, idx, S):
        res = super().decode(idx)
        return self.to4D(res, S)

if __name__ == "__main__":
    import time
    from framework.evaluate import *

    X = cv2.imread("/Users/alex/Desktop/proj/compression/data/Kodak/kodim01.png", 0)
    X = X.reshape(1, X.shape[0], X.shape[1], 1)
    t0 = time.time()
    vq = cwVQ4D(cw_idx=[0, 1, 2, 3, 4],
               cw_N=[6, 7, 7, 7],
               PSNR_TH=30,
               win=2)
    vq.fit(X)
    idx, S = vq.encode(copy.deepcopy(X))
    iX = vq.decode(idx, S)
    print('   \033[37m-->cwVQ using %s codewords, PSNR=%f, using time %5f sec'%(str(vq.cw_N), PSNR(X, iX), time.time()-t0))
    t0 = time.time()
    km = kmVQ4D(np.sum(vq.cw_N), 2)
    km.fit(X)
    idx, S = km.encode(X)
    iX = km.decode(idx, S)
    print('   -->VQ using %d codewords, PSNR=%f, using time %5f sec\033[0m'%(np.sum(vq.cw_N), PSNR(X, iX), time.time()-t0))
