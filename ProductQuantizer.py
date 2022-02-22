# 2022.01.21
import faiss

class ProductQuantizer():
    def __init__(self, n_codes, code_size=1):
        self.log_n_codes = (int)(np.log2(n_codes-1))+1
        self.n_codes = pow(2, self.log_n_codes)
        self.code_size = code_size
        self.dim = -1
        self.codebook = None
        
    def fit(self, X):
        X = X.reshape(-1, X.shape[-1])
        self.dim = X.shape[-1]
        pq = faiss.ProductQuantizer(self.dim, self.code_size, self.log_n_codes)
        pq.train(X)
        self.codebook = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
        return self
    
    def predict(self, X):
        S = (list)(X.shape)
        S[-1] = -1
        X = X.reshape(-1, X.shape[-1])
        pq = faiss.ProductQuantizer(self.dim, self.code_size, self.log_n_codes)
        faiss.copy_array_to_vector(self.codebook.ravel(), pq.centroids)
        codes = pq.compute_codes(X)
        return codes.reshape(S)
    
    def inverse_predict(self, codes):
        S = (list)(codes.shape)
        S[-1] = -1
        codes = codes.reshape(-1, codes.shape[-1])
        pq = faiss.ProductQuantizer(self.dim, self.code_size, self.log_n_codes)
        faiss.copy_array_to_vector(self.codebook.ravel(), pq.centroids)
        X = pq.decode(codes)
        return X.reshape(S)