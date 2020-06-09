# 2020.06.05
# active learning: query by committee
# modified from Xiou
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import time

class QBC():
    def __init__(self, learners, init=0.01, n_increment=200, n_iter=40, percent=0.05):
        self.init = init
        self.n_increment = n_increment
        self.n_learner = len(learners)
        self.n_iter = n_iter
        self.num_class = 3
        self.learners = learners
        self.percent = percent
        self.trained = False
        self.acc_t = []
        self.acc_v = []
    
    def metric(self, prob):
        return entropy(prob, base=self.num_class, axis=1)

    def fit(self, x, y, xv=None, yv=None):
        self.trained = True
        self.num_class = np.unique(y).shape[0]
        #x, xt, y, yt = train_test_split(x, y, train_size=self.init, random_state=42, stratify=y)
        idx = np.random.choice(x.shape[0], (int)(x.shape[0]*self.percent))
        x_train, y_train = x[idx], y[idx]
        x_pool = np.delete(x, idx, axis=0)
        y_pool = np.delete(y, idx, axis=0)
        acc_t, acc_v, s = [], [], []
        for k in range(self.n_iter):
            print('       start iter -> %3s'%str(k))
            t0 = time.time()
            for i in range(self.n_learner):
                self.learners[i].fit(x_train, y_train)
            pt = self.predict_proba(x_pool)
            at = accuracy_score(y_pool, np.argmax(pt, axis=1))
            acc_t.append(at)
            s.append(y_pool.shape[0])
            ht = self.metric(pt)
            try:
                xv.shape
                print('           test shape: %s, val shape: %s'%(str(x_pool.shape), str(xv.shape)))
                pv = self.predict_proba(xv)
                av = accuracy_score(yv, np.argmax(pv, axis=1))
                print('           <Acc> test: %s, val: %s'%(at, av))
                acc_v.append(av)
                hv = self.metric(pv)
                print('           <Entropy> test: %s, val: %s'%(np.mean(ht), np.mean(hv)))
            except:
                pass
            idx = np.argsort(ht)[-self.n_increment:]
            x_train = np.concatenate((x_train, x_pool[idx]), axis=0)
            y_train = np.concatenate((y_train, y_pool[idx]), axis=0)
            x_pool = np.delete(x_pool, idx, axis=0)
            y_pool = np.delete(y_pool, idx, axis=0)
            print('       end iter -> %3s using %10s seconds\n'%(str(k),str(time.time()-t0)))
        self.acc_t = acc_t
        self.acc_v = acc_v
        return s, acc_t, acc_v

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
    import pickle
    import sys
    #path = '/mnt/yifan/face/'
    path = '../../fea/'
    i = (int)(sys.argv[1])
    with open(path+str(i)+'_discard.pkl', 'rb') as f:
        d = pickle.load(f)
    X_train, y_train, X_test, y_test = d['x'],d['y'],d['xt'],d['yt']

    clf = QBC(init=0.05, n_increment=200, n_iter=14,
              learners=[SVC(gamma='auto', probability=True)])
    s, a, b = clf.fit(X_train, y_train, X_test, y_test)
    clf = SVC(gamma='auto', probability=True)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    save = {'shape':s, 'train':a, 'test':b}
    with open('qbc_0327'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(save, f)

        
   