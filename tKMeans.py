# v2020.03.18
# multi classifier
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from myRegression import myRegression

class tKMeans():
    def __init__(self, X, Y, MSE=10, learner=sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='ovr', n_jobs=20), percent=0.05, split_trial=10):
        self.data = {}
        self.MSE = MSE
        self.num_sample = X.shape[0]*percent
        self.num_class = np.unique(Y).shape[0]
        self.learner = learner
        self.percent = percent
        self.predict_label = []
        self.split_trial = split_trial
        kmeans = KMeans(n_clusters=2, n_jobs=10, init='k-means++').fit(X)
        pred = kmeans.predict(X)
        self.data['KMeans'] = kmeans
        for i in range(2):
            self.data[str(i)] = {'X':X[pred == i], 'Y':Y[pred == i]}

    def calMSE(self, nX):
        nMSE = sklearn.metrics.pairwise.euclidean_distances(nX, np.mean(nX, axis=0).reshape(1,-1))
        #print(np.mean(nMSE))
        return nMSE
        
    def check_split(self):
        k = self.data.copy()
        key = k.keys()
        for k in key:
            if k == 'KMeans':
                continue
            if self.data[k]['X'].shape[0] > self.num_sample and np.mean(self.calMSE(self.data[k]['X'])) > self.MSE:
                kmeans = KMeans(n_clusters=2, n_jobs=10, init='k-means++').fit(self.data[k]['X'])
                pred = kmeans.predict(self.data[k]['X'])
                for i in range(2):
                    self.data[k+str(i)] = {'X':self.data[k]['X'][pred == i], 'Y':self.data[k]['Y'][pred == i]}
                self.data[k]['X'] = np.array([1])
                self.data[k]['Y'] = None
                self.data[k]['KMeans'] = kmeans
    
    def train(self):
        for k in self.data.keys():
            if k == 'KMeans':
                continue
            if self.data[k]['X'].shape[0] < 2 :
                continue
            reg = myRegression(self.learner, self.num_class)
            reg.fit(self.data[k]['X'], self.data[k]['Y'])
            reg.score(self.data[k]['X'], self.data[k]['Y'])
            self.data[k]['learner'] = reg
            
    def fit(self, X, Y):
        for i in range(self.split_trial):
            self.check_split()
        self.train()

    def test_tree(self, key, X):
        if 'KMeans' in self.data[key].keys():
            npred = self.data[key]['KMeans'].predict(X)
            self.test_tree(key+str(npred[0]), X)
        else:
            pred = self.data[key]['learner'].predict(X)
            self.predict_label.append(pred[0])
        
    def predict(self, X):
        self.predict_label = []
        for i in range(X.shape[0]):
            pred = self.data['KMeans'].predict(X[i].reshape(1,-1))
            self.test_tree(str(pred[0]), X[i].reshape(1,-1))
        self.predict_label = np.array(self.predict_label)
        return self.predict_label
    
    def score(self, X, Y):
        self.predict(X)
        return accuracy_score(Y, self.predict_label)
            
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    print(" \n> This is a test enample: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    print(" input feature shape: %s"%str(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2,  stratify=digits.target)
    
    clf = tKMeans(X_train, y_train, MSE=20, learner=LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr', n_jobs=20, max_iter=1000))
    clf.fit(X_train, y_train)
    print(" --> train acc: %s"%str(clf.score(X_train, y_train)))
    print(" --> test acc.: %s"%str(clf.score(X_test, y_test)))
    print("------- DONE -------\n")