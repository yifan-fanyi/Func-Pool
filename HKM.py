# @yifan
# 2021.12.05
# Hierarchical VQ
# return label and depth 
import numpy as np
import faiss
from myKMeans import myKMeans
from evaluate import MSE

class HKM():
    def __init__(self, n_nodes_p_level):
        self.n_nodes_p_level = n_nodes_p_level
        self.depth = len(n_nodes_p_level)
        self.init_cluster_centers_ = None
        self.cluster_centers_ = {}
        self.mapping = {}
        self.TH = -1

    def fit_fast(self, init_cluster_centers_):
        self.init_cluster_centers_ = init_cluster_centers_
        self.cluster_centers_['L-'+str(self.depth-1)] = self.init_cluster_centers_
        for i in range(self.depth-2, -1, -1):
            km = myKMeans(n_clusters=self.n_nodes_p_level[i])
            km.fit(self.cluster_centers_['L-'+str(i+1)])
            l = km.predict(self.cluster_centers_['L-'+str(i+1)]).reshape(-1)
            self.cluster_centers_['L-'+str(i)] = km.cluster_centers_
            mp = {}
            for j in range(len(l)):
                if l[j] in mp.keys():
                    mp[l[j]].append(j)
                else:
                    mp[l[j]] = [j]
            self.mapping['L-'+str(i)+'~L-'+str(i+1)] = mp
        return self
    
    def fit(self, X):
        for i in range(self.depth):
            km = myKMeans(n_clusters=self.n_nodes_p_level[i])
            km.fit(X)
            self.cluster_centers_['L-'+str(i)] = km.cluster_centers_
        for i in range(1, self.depth):
            l = self.Cpredict(self.cluster_centers_['L-'+str(i)], self.cluster_centers_['L-'+str(i-1)]).reshape(-1)
            mp = {}
            for j in range(len(l)):
                if l[j] in mp.keys():
                    mp[l[j]].append(j)
                else:
                    mp[l[j]] = [j]
            self.mapping['L-'+str(i-1)+'~L-'+str(i)] = mp
        return self

    def Cpredict(self, X, cent):
        index = faiss.IndexFlatL2(cent.shape[1]) 
        index.add(cent)             
        D, I = index.search(X, 1)
        return I
    
    def predict(self, X, TH):
        self.TH = TH
        S = (list)(X.shape)
        S[-1] = -1
        X = X.reshape(-1, X.shape[-1])
        depth, label = np.zeros(X.shape[0])-1, np.zeros(X.shape[0])-1
        tmp0 = self.Cpredict(X, self.cluster_centers_['L-'+str(0)]).reshape(-1)
        iX0 = self.cluster_centers_['L-'+str(0)][tmp0]
        n0 = self.cluster_centers_['L-'+str(0)].shape[0]
        for i in range(1, self.depth):
            tmp = self.Cpredict(X, self.cluster_centers_['L-'+str(i)]).reshape(-1)
            iX = self.cluster_centers_['L-'+str(i)][tmp]
            n = self.cluster_centers_['L-'+str(i)].shape[0]
            for j in range(X.shape[0]):
                if label[j] < 0:
                    a, b = MSE(X[j], iX0[j]) - MSE(X[j], iX[j]), np.log2(n)  - np.log2(n0)
                    #print('   dMSE=%3.4f, dbpp=%3.4f, dMSE/dbpp=%3.4f'%(a,b,a/b))
                    if a / b < self.TH:
                        label[j] = tmp[j]
                        depth[j] = i
            tmp0, iX0, n0 = tmp, iX, n
        for j in range(X.shape[0]):
            if label[j] < 0 or depth[j] < 0:
                label[j] = tmp[j]
                depth[j] = i
        return label.reshape(S).astype('int16'), depth.reshape(S)

    def inverse_predict(self, label, depth):
        S = (list)(label.shape)
        S[-1] = -1
        label, depth = label.reshape(-1), depth.reshape(-1)
        iX = np.zeros((label.shape[0], self.cluster_centers_['L-'+str(0)].shape[-1]))
        for i in range(self.depth):
            idx = (depth == i)
            iX[idx] = self.cluster_centers_['L-'+str(i)][label[idx]]
        return iX.reshape(S)

        