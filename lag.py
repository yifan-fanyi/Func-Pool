# v2020.03.20
# label assistant regression
import numpy as np
import time
import scipy
import keras
from sklearn import preprocessing 
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

class LAG():
    def __init__(self, encode='onehot', num_clusters=[10,10], alpha=5, par={}, learner=None):
        self.encode = encode 
        self.num_clusters = num_clusters 
        self.alpha = alpha
        self.learner = learner
        self.clus_labels = []
        self.centroid = []
        
    def compute_target_(self, X, Y, batch_size): 
        Y = Y.reshape(-1)
        class_list = np.unique(Y)
        labels = np.zeros((X.shape[0]))
        self.clus_labels = np.zeros((np.sum(np.array(self.num_clusters)),))
        self.centroid = np.zeros((np.sum(np.array(self.num_clusters)), X.shape[1]))
        start = 0
        for i in range(len(class_list)):
            ID = class_list[i]
            feature_train = X[Y==ID]
            if batch_size == None:
                kmeans = KMeans(n_clusters=self.num_clusters[i], verbose=0, random_state=9, n_jobs=10).fit(feature_train)
            else:
                kmeans = MiniBatchKMeans(n_clusters=self.num_clusters[i], verbose=0, batch_size=batch_size, n_init=5).fit(feature_train)
            labels[Y==ID] = kmeans.labels_ + start
            self.clus_labels[start:start+self.num_clusters[i]] = ID
            self.centroid[start:start+self.num_clusters[i]] = kmeans.cluster_centers_
            start += self.num_clusters[i]
        return labels

    def fit(self, X, Y, batch_size=None):
        labels_train = self.compute_target_(X, Y, batch_size=batch_size)    
        if self.encode == 'distance':
            labels_train_onehot = np.zeros((labels_train.shape[0], self.clus_labels.shape[0]))
            for i in range(labels_train.shape[0]):
                gt = Y[i].copy()
                dis = euclidean_distances(X[i].reshape(1,-1), self.centroid[self.clus_labels == gt]).reshape(-1)
                dis = dis / (dis.min() + 1e-15)
                p_dis = np.exp(-dis*self.alpha)
                p_dis = p_dis / p_dis.sum()
                labels_train_onehot[i, self.clus_labels == gt] = p_dis            
        elif self.encode == 'onehot':
            labels_train_onehot = keras.utils.to_categorical(labels_train, np.unique(labels_train).shape[0])     
        else:
            print("       <Warning>        Using raw label for LLSR.")
            labels_train_onehot = labels_train

        self.learner.fit(X, labels_train_onehot)
        
    def predict(self, X):
        return self.learner.predict(X)
    
    def predict_proba(self, X):
        return self.learner.predict_proba(X)
    
    def score(self, X, Y):
        X = self.predict_proba(X)
        pred_labels = np.zeros((X.shape[0], len(np.unique(Y))))
        for km_i in range(len(np.unique(Y))):
            pred_labels[:,km_i] = X[:, self.clus_labels==km_i].sum(1)
        pred_labels = np.argmax(pred_labels, axis=1)
        idx = pred_labels == Y.reshape(-1)
        return 1. * np.count_nonzero(idx) / Y.shape[0]

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    print(" \n> This is a test enample: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    print(" input feature shape: %s"%str(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2,  stratify=digits.target)

    from llsr import LLSR
    clf = LAG(encode='distance', num_clusters=[2,2,2,2,2,2,2,2,2,2], alpha=5, learner=LLSR(onehot=False))  
    clf.fit(X_train, y_train)
    print(" --> train acc: %s"%str(clf.score(X_train, y_train)))
    print(" --> test acc.: %s"%str(clf.score(X_test, y_test)))
    print("------- DONE -------\n")