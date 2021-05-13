# 2021.05.10
# change inv_predict to inverse_perdict
# https://www.kdnuggets.com/2021/01/k-means-faster-lower-error-scikit-learn.html
# faster kmeans
# gpu is supported
# conda install -c conda-forge faiss-gpu
#
import faiss
import numpy as np

class fast_KMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, gpu=False, n_threads=10):
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
                                    )
        else:
            self.kmeans = faiss.Kmeans(d=X.shape[1],
                                    k=self.n_clusters,
                                    niter=self.max_iter,
                                    nredo=self.n_init,
                                    )
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