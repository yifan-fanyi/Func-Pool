# split one indices into two and encode them using two Huffman
# the performance is upper bounded by single Huffman
# not used
# 2021.09.23
# x
import numpy as np
import copy
from myKMeans import myKMeans, Mapping
from Huffman import Huffman

class ConditionalEntropy():
    def __init__(self, n_groups, cluster_center):
        self.n_groups = n_groups
        self.KM = myKMeans(n_clusters=n_groups, saveObj=True)
        self.Map = []#self.get_map(cluster_center)
        self.H1 = Huffman()
        self.H2 = Huffman()
        self.Mp_list = []
        self.onlyone = []

    def get_map(self, cluster_center):
        self.KM.fit(cluster_center)
        idx = self.KM.predict(cluster_center).reshape(-1)
        mp, inv_mp = {}, {}
        for i in range(len(idx)):
            mp[i] = idx[i]
            inv_mp[idx[i]] = i
        return Mapping(Cent=None, mp=mp, imp=inv_mp)
    
    def get_map1(self, idx):
        l = len(np.unique(idx))
        iv = l / self.n_groups
        mp, inv_mp = {}, {}
        for i in range(l):
            mp[i] = (int)(i / iv)
            inv_mp[(int)(i / iv)] = i
        return Mapping(Cent=None, mp=mp, imp=inv_mp)

    def sort_map(self, idx):
        tmp = {}
        for i in range(len(idx)):
            if idx[i] in tmp.keys():
                tmp[idx[i]] += 1
            else:
                tmp[idx[i]] = 1
        sort_tmp = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1], reverse=True)}
        mp, imp = {}, {}
        ct = 0
        for i in sort_tmp:
            mp[i] = ct
            imp[ct] = i
            ct += 1
        return Mapping(mp, imp)

    def fit(self, idx):
        self.Map = self.get_map1(idx)
        idx = idx.reshape(-1)
        idx1 = self.Map.transform(copy.deepcopy(idx))
        self.H1.fit(copy.deepcopy(idx1))
        idx2 = np.zeros_like(idx)
        for i in range(self.n_groups):
            f = idx1 == i
            mp = self.sort_map(copy.deepcopy(idx[f]))
            self.Mp_list.append(mp)
            idx2[f] = self.Mp_list[i].transform(copy.deepcopy(idx[f]))
            if len(np.unique(copy.deepcopy(idx[f]))) > 1:
                self.onlyone.append(False)
            else:
                self.onlyone.append(True)
        self.H2.fit(idx2)
        return self
    
    def encode(self, idx):
        self.S = idx.shape
        idx = idx.reshape(-1)
        idx1 = self.Map.transform(copy.deepcopy(idx))
        idx2 = np.zeros_like(idx)
        for i in range(self.n_groups):
            f = idx1 == i
            idx2[f] = self.Mp_list[i].transform(copy.deepcopy(idx[f]))
        stream = ''
        for i in range(len(idx2)):
            stream += self.H1.encode(idx1[i])
            if self.onlyone[idx1[i]] == False:
                stream += self.H2.encode(idx2[i])
        return stream
    
    def decode(self, stream, S=None):
        if S is not None:
            self.S = S 
        idx = np.zeros(self.S).reshape(-1)
        idx1 = np.zeros(self.S).reshape(-1)
        idx2 = np.zeros(self.S).reshape(-1)
        last = 0
        for i in range(len(idx2)):
            val, last = self.H1.decode(stream, last, 1)
            idx1[i] = val[0]
            if self.onlyone[(int)(idx1[i])] == False:
                val, last = self.H2.decode(stream, last, 1)
                idx2[i] = val[0]
        for i in range(self.n_groups):
            f = idx1 == i
            idx[f] = self.Mp_list[i].inverse_transform(copy.deepcopy(idx2[f]))
        return idx.reshape(S)


if __name__ == "__main__":
    from scipy.stats import entropy
    idx = np.random.poisson(4, size=(10000))
    hist = np.zeros(len(np.unique(idx)))
    for i in range(len(idx)):
        hist[idx[i]] +=1
    hist /= len(idx)
    print("Entropy =", entropy(hist, base=2))

    print(np.min(idx), np.max(idx), len(np.unique(idx)))

    hf = Huffman().fit(idx)
    stream = hf.encode(idx)
    print('Huffman avg =', len(stream)/(idx.shape[0]))

    ce = ConditionalEntropy(2, None)
    ce.fit(idx)
    stream = ce.encode(idx)
    print('Conditional Entropy avg =', len(stream)/(idx.shape[0]))