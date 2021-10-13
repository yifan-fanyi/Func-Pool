import numpy as np
from collections import Counter

class KNN():
    def __init__(self, k, distance_function):
        self.k = k
        self.distance_function = distance_function
        self.__version__ = "2021.07.04"
        
    def get_k_neighbors(self, point):
        dist = []
        for train in self.features:
            dist.append(self.distance_function(train, point))
        return np.argsort(dist)[:self.k]
    
    def get_k_neighbours_label(self, point):
        idx = self.get_k_neighbors(point)
        return (list)(np.array(self.labels)[idx])
    
    def fit(self, features, labels=None):
        self.features = features
        self.labels = labels
        return self
    
    def getKNN(self, features):
        pred = []
        for test in features:
            knn = self.get_k_neighbours(test)
            pred.append(knn)
        return pred 
    
    def predict(self, features):
        pred = []
        for test in features:
            k_labels = self.get_k_neighbours_label(test)
            counter = Counter(k_labels)
            pred.append(counter.most_common(1)[0][0])
        return pred