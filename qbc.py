# 2020.03.20
# active learning: query by committee
# modified from Xiou
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import time

class QBC():
    def __init__(self, learners, init=0.01, n_increment=200, n_iter=20, percent=0.5):
        self.init = init
        self.n_increment = n_increment
        self.n_learner = len(learners)
        self.n_iter = n_iter
        self.num_class = 3
        self.learners = learners
        self.percent = percent
        self.trained = False
    
    def metric(self, prob):
        return entropy(prob, base=self.num_class, axis=1)

    def fit(self, x, y, xv=None, yv=None):
        self.trained = True
        self.num_class = np.unique(y).shape[0]
        x, xt, y, yt = train_test_split(x, y, train_size=self.init, random_state=42, stratify=y)
        acc_t, acc_v = [], []
        for k in range(self.n_iter):
            print('       start iter -> %3s'%str(k))
            t0 = time.time()
            for i in range(self.n_learner):
                idx = np.random.choice(x.shape[0], (int)(x.shape[0]*self.percent))
                self.learners[i].fit(x[idx], y[idx])
            pt = self.predict_proba(xt)
            at = accuracy_score(yt, np.argmax(pt, axis=1))
            acc_t.append(at)
            ht = self.metric(pt)
            try:
                xv.shape
                print('           test shape: %s, val shape: %s'%(str(xt.shape), str(xv.shape)))
                pv = self.predict_proba(xv)
                av = accuracy_score(yv, np.argmax(pv, axis=1))
                print('           <Acc> test: %s, val: %s'%(at, av))
                acc_v.append(av)
                hv = self.metric(pv)
                print('           <Entropy> test: %s, val: %s'%(np.mean(ht), np.mean(hv)))
            except:
                pass
            idx = np.argsort(ht)[-self.n_increment:]
            x = np.concatenate((x, xt[idx]), axis=0)
            y = np.concatenate((y, yt[idx]), axis=0)
            xt = np.delete(xt, idx, axis=0)
            yt = np.delete(yt, idx, axis=0)
            print('       end iter -> %3s using %10s seconds\n'%(str(k),str(time.time()-t0)))
        return self

    def predict_proba(self, x):
        assert (self.trained == True), "Must call fit first!"
        pred = np.zeros((x.shape[0], self.num_class))
        for i in range(self.n_learner):
            pred += self.learners[i].predict_proba(x)
        return pred / np.sum(pred, axis=1, keepdims=True)
    
    def predict(self, x):
        assert (self.trained == True), "Must call fit first!"
        pred = self.predict_proba(x)
        return np.argmax(pred, axis=1)
    
    def score(self, x, y):
        assert (self.trained == True), "Must call fit first!"
        pred = self.predict(x)
        return accuracy_score(y, pred)

if __name__ == "__main__":
    from sklearn.svm import SVC
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from mylearner import myLearner
    
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    print(" input feature shape: %s"%str(X.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2,  stratify=digits.target)
    
    clf = QBC(init=0.01, n_increment=40, n_iter=20,
              learners=[myLearner(SVC(gamma='scale', probability=True), len(np.unique(y_train))),
                        myLearner(SVC(gamma='scale', probability=True), len(np.unique(y_train))),
                        myLearner(SVC(gamma='scale', probability=True), len(np.unique(y_train)))])
    clf.fit(X_train, y_train, X_test, y_test)
    print(" --> train acc: %s"%str(clf.score(X_train, y_train)))
    print(" --> val acc: %s"%str(clf.score(X_test, y_test)))
    print("------- DONE -------\n")

        
   