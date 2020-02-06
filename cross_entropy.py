# v2020.02.05
#
# class Cross_Entropy()
#   Reference:
#         Manimaran A, Ramanathan T, You S, et al. Visualization, Discriminability and Applications of Interpretable Saak Features[J]. 2019.
#   Compute cross entropy across each feature for feature selection
#     input:
#         x     -> (n, d)
#         y     -> (n,1)
#     return
#               -> (d)   
#
# KMeans_Cross_Entropy
# ML_Cross_Entropy

import numpy as np 
import math
import sklearn
import keras
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from regression import * 

class Cross_Entropy():
    def __init__(self, num_class, num_bin=10):
        self.num_class = (int)(num_class)
        self.num_bin = (int)(num_bin)

    def bin_process(self, x ,y):
        if np.max(x) ==  np.min(x):
            return -1*np.ones(self.num_bin)
        x = ((x - np.min(x)) / (np.max(x) - np.min(x))) * (self.num_bin)
        mybin = np.zeros((self.num_bin, self.num_class))
        b = x.astype('int64')
        b[b == self.num_bin] -= 1
        mybin[b,y] += 1.
        for l in range(0,self.num_class):
            p = np.array(y[ y==l ]).shape[0]
            mybin[:,l] /= (float)(p)
        return np.argmax(mybin, axis=1)
    
    def kmeans_process(self, x, y):
        kmeans = KMeans(n_clusters=self.num_bin, random_state=0).fit(x.reshape(1,-1))
        mybin = np.zeros((self.num_bin, self.num_class))
        b = kmeans.labels_.astype('int64')
        b[b == self.num_bin] -= 1
        mybin[b,y] += 1.
        for l in range(0,self.num_class):
            p = np.array(y[ y==l ]).shape[0]
            mybin[:,l] /= (float)(p)
        return np.argmax(mybin, axis=1)

    def compute_prob(self, x, y):
        prob = np.zeros((self.num_class, x.shape[1]))
        for k in range(0, x.shape[1]):
            mybin = self.bin_process(x[:,k], y[:,0])
            #mybin = self.kmeans_process(x[:,k], y[:,0])
            for l in range(0, self.num_class):
                p = mybin[mybin == l]
                p = np.array(p).shape[0]
                prob[l,k] = p / (float)(self.num_bin)
        return prob

    def compute(self, x, y, class_weight=None):
        x = x.astype('float64')
        y = y.astype('int64')
        prob = self.compute_prob(x, y)
        prob = -1 * np.log10(prob + 1e-5) / np.log10(self.num_class)
        y = np.moveaxis(y, 0, 1)
        H = np.zeros((self.num_class, x.shape[1]))
        for c in range(0, self.num_class):
            yy = y == c
            p = prob[c].reshape(prob.shape[1], 1)
            p = p.repeat(yy.shape[1], axis=1)
            H[c] += np.mean(yy * p, axis=1)
        if class_weight != None:
            class_weight = np.array(class_weight)
            H *= class_weight.reshape(class_weight.shape[0],1) * self.num_class
        return np.sum(H, axis=0)

# new cross entropy
def KMeans_Cross_Entropy(X, Y, num_class, num_bin=32):
    if np.unique(Y).shape[0] == 1: #alread pure
        return 0
    if X.shape[0] < num_bin:
        return -1
    kmeans = KMeans(n_clusters=num_bin, random_state=0).fit(X)
    prob = np.zeros((num_bin, num_class))
    for i in range(num_bin):
        idx = (kmeans.labels_ == i)
        tmp = Y[idx]
        for j in range(num_class):
            prob[i, j] = (float)(tmp[tmp == j].shape[0]) / ((float)(Y[Y==j].shape[0]) + 1e-5)
    prob = (prob)/(np.sum(prob, axis=1).reshape(-1,1) + 1e-5)
    true_indicator = keras.utils.to_categorical(Y, num_classes=num_class)
    probab = prob[kmeans.labels_]
    return sklearn.metrics.log_loss(true_indicator,probab)/math.log(num_class)

# new machine learning based cross entropy
def ML_Cross_Entropy(X, Y, num_class):
    X, XX, Y, YY = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=42, stratify=Y)
    reg = myRegression(RandomForestClassifier(n_estimators=100, max_depth=7, verbose=0, n_jobs=-1, class_weight='balanced'),
                        num_class)
    reg.fit(X, Y)
    pred = reg.predict_proba(XX)
    pred = pred[YY.reshape(-1)]
    reg.score(X, Y)
    reg.score(XX, YY)
    true_indicator = keras.utils.to_categorical(YY, num_classes=num_class)
    return sklearn.metrics.log_loss(true_indicator, pred)/math.log(num_class)

def CE(X, Y, num_class, mode=1):
    H = []
    for i in range(X.shape[1]):
        if mode == 1:
            H.append(KMeans_Cross_Entropy(X[:, i].reshape(-1, 1), Y, num_class, num_bin=32))
        elif mode ==1:
            H.append(ML_Cross_Entropy(X[:, i].reshape(-1, 1), Y, num_class))     
    return np.array(H)

if __name__ == "__main__":
    import cv2
    X = cv2.imread('./data/se.jpg')
    cv2.imwrite('./data/se.jpg', X)
    Y = cv2.imread('./data/gt.jpg', 0)
    Y[Y!=0] = 1
    X = X.reshape(-1, 3)
    Y = Y.reshape(-1, 1)
    print(CE(X, Y, 2))

    '''
    import time
    t0 = time.time()
    x = np.array([1,2,3,1,3,5,7,1,2,4])
    y = np.array([0,0,1,0,1,0,1,0,1,1])
    ce = Cross_Entropy(2)
    H = ce.compute(x.reshape(10,1), y.reshape(10,1), class_weight=None)
    print(H)
    print('ideal: ', str([1.12576938]))
    print('Using time: ', time.time()-t0)
    '''