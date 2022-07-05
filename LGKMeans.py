
from calendar import c
import faiss
import numpy as np
import copy
import sklearn
import pickle
from sklearn import cluster
import time

def Cpredict(X, cent):
    X = np.ascontiguousarray(X.astype('float32'))
    cent = np.ascontiguousarray(cent.astype('float32'))
    index = faiss.IndexFlatL2(cent.shape[1]) 
    index.add(cent)             
    _, I = index.search(X, 1)
    return I

class fast_KMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, gpu=False, n_threads=16):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.gpu = gpu     
        faiss.omp_set_num_threads(n_threads)       
        self.__version__ = faiss.__version__

    def fit(self, X):
        if self.gpu != False:
            self.kmeans = faiss.Kmeans(d=X.shape[1],
                                    k=self.n_clusters,
                                    niter=self.max_iter,
                                    nredo=self.n_init,
                                    gpu=self.gpu,
                                    min_points_per_centroid=1,
                                    max_points_per_centroid=1024)
        else:
            self.kmeans = faiss.Kmeans(d=X.shape[1],
                                    k=self.n_clusters,
                                    niter=self.max_iter,
                                    nredo=self.n_init,
                                    min_points_per_centroid=1,
                                    max_points_per_centroid=1024)
        X = np.ascontiguousarray(X.astype('float32'))
        self.kmeans.train(X)
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]
        return self
        
    def predict(self, X):
        X = np.ascontiguousarray(X.astype('float32'))
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]
    
    def inverse_predict(self, label):
        return self.cluster_centers[label]

class ClusterNode:
    def __init__(self, dim):
        self.dim = dim
        self.vec_sum = np.zeros(dim).astype('float64')
        self.vec_ct = 0.

    def _get_centroid(self):
        return self.vec_sum / self.vec_ct
    
    def _removeVec(self, vec):
        self.vec_sum -= vec
        self.vec_ct -= 1
    
    def _addVec(self, vec):
        self.vec_sum += vec
        self.vec_ct += 1

    def _removeVecs(self, vecs):
        self.vec_sum -= np.sum(vecs, axis=0)
        self.vec_ct -= vecs.shape[0]
    
    def _addVecs(self, vecs):
        self.vec_sum += np.sum(vecs, axis=0)
        self.vec_ct += vecs.shape[0]

def _file_opener(file):
    with open(file,'rb') as f:
        return pickle.load(f)

def _file_writer(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f, 4)

class LGKMeans:
    def __init__(self, n_clusters, max_iter, stop_diff=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.dim = -1
        self.nodes = []
        self.last_centroids = None
        self.stop_diff = stop_diff

    def _init(self, files):
        for i in range(len(files)):
            file = files[i]
            X = _file_opener(file)
            if i == 0:
                self.dim = X.shape[1]
                km = fast_KMeans(n_clusters=self.n_clusters).fit(X)
            label = km.predict(X).reshape(-1)
            _file_writer(file+'_label', label)
            for j in range(self.n_clusters):
                idx = label == j
                if i == 0:
                    new_node = ClusterNode(X.shape[1])
                    new_node._addVecs(X[idx])
                    self.nodes.append(new_node)
                else:
                    self.nodes[j]._addVecs(X[idx])
        return self

    def _get_centroids(self):
        centroids = []
        for i in range(self.n_clusters):
            centroids.append(self.nodes[i]._get_centroid())
        return np.array(centroids)

    def _iter(self, files):
        centroids = self._get_centroids()
        for file in files:
            X = _file_opener(file)
            last_label = _file_opener(file+'_label')
            label = Cpredict(X, centroids).reshape(-1)
            _file_writer(file+'_label', label)
            for j in range(self.n_clusters):
                idx = np.logical_and(label != j, last_label == j) 
                self.nodes[j]._removeVecs(X[idx])
                idx = np.logical_and(label == j, last_label != j) 
                self.nodes[j]._addVecs(X[idx])
        if self.last_centroids is None:
            change = 1000
        else:
            change = np.mean(np.square(self.last_centroids - centroids))
        self.last_centroids = centroids
        return change

    def _stop(self, change):
        if change <= self.stop_diff:
            return True
        return False

    def fit(self, files):
        self._init(files)
        t0 = time.time()
        for i in range(self.max_iter):
            change = self._iter(files)
            print('   iter %d -> time=%f, diff=%f'%(i, time.time()-t0, change))
            t0 = time.time()
            if self._stop(change) == True:
                break

        return self
        
if __name__ == "__main__":
    
    files = []
    X = _file_opener('clic_test.pkl')
    for i in range(186):
        _file_writer('./data/'+str(i)+'.pkl', X[i:i+1].reshape(-1, 3))
        files.append('./data/'+str(i)+'.pkl')


    km = LGKMeans(n_clusters=32, max_iter=128, stop_diff=0.01)
    km.fit(files)
    cent = km._get_centroids()
    print(cent)
    X = _file_opener('clic_test.pkl')
    X = X.reshape(-1, 3)
    label = Cpredict(X, cent)
    iX = cent[label.reshape(-1)]
    print(np.mean(np.square(X-iX)))

    print('\nfast_KMeans')
    X = _file_opener('clic_test.pkl')
    X = X.reshape(-1, 3)
    fm = fast_KMeans(32, max_iter=128).fit(X)
    cent = fm.cluster_centers_
    print(cent)
    label = Cpredict(X, cent)
    iX = cent[label.reshape(-1)]
    print(np.mean(np.square(X-iX)))


        