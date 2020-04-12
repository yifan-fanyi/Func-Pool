# 2020.04.11
import numpy as np 
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from mylearner import myLearner

class HierNode():
    def __init__(self, learner, num_class, num_cluster, metric, isleaf=False, id='R'):
        self.learner = myLearner(learner=learner, num_class=num_class)
        self.kmeans = KMeans(n_clusters=num_cluster)
        self.num_cluster = num_cluster
        self.metric = metric
        self.isleaf = isleaf
        self.id = id

    def metric_(self):
        # define the spliting criteria here
        # return is current node a leafnode
        return False

    def fit(self, X, Y):
        self.kmeans.fit(X)
        if self.metric_() == True:
            self.isleaf = True
        if self.isleaf == True:
            self.learner.fit(X, Y)
    
    def predict(self, X):
        if self.isleaf == True:
            try:
                prob = self.learner.predict_proba(X)
            except:
                prob = self.learner.predict(X)
            return prob
        else:
            return self.kmeans.predict(X)

class HierKmeans():
    def __init__(self, depth, learner, num_cluster, metric):
        self.nodes = {}
        self.depth = depth
        self.learner = learner
        self.num_cluster = num_cluster
        self.metric = metric
        self.num_class = -1
        self.trained = False

    def fit(self, X, Y):
        self.num_class = len(np.unique(Y))
        tmp_data = [{'X':X, 'Y':Y, 'id':'R'}]
        for i in range(self.depth):
            tmp = []
            for j in range(len(tmp_data)):
                tmp_node = HierNode(learner=self.learner, 
                                    num_class=self.num_class, 
                                    num_cluster=self.num_cluster, 
                                    metric=self.metric, 
                                    isleaf=(i==self.depth-1), 
                                    id=tmp_data[j]['id'])

                tmp_node.fit(tmp_data[j]['X'], tmp_data[j]['Y'])
                label = tmp_node.predict(tmp_data[j]['X'])
                self.nodes[tmp_data[j]['id']] = copy.deepcopy(tmp_node)
                if tmp_node.isleaf == True:
                    continue
                for k in range(self.num_cluster):
                    idx = (label == k)
                    if idx.shape[0] == 0:
                        continue
                    tmp.append({'X':tmp_data[j]['X'][idx], 'Y':tmp_data[j]['Y'][idx], 'id':tmp_data[j]['id']+str(k)})
            if len(tmp) == 0 and i != self.depth-1:
                print("       <Warning> depth %s not achieved, actual depth %s"%(str(self.depth), str(i+1)))
                self.depth = i
                break
            tmp_data = tmp
        self.trained = True
    
    def predict_proba(self, X):
        assert (self.trained == True), "Must call fit first!"
        tmp_pred = []
        tmp_data = [{'X':X, 'idx':np.arange(0, X.shape[0]), 'id':'R'}]
        for i in range(self.depth):
            tmp = []
            for j in range(len(tmp_data)):
                if self.nodes[tmp_data[j]['id']].isleaf == True:
                    prob = self.nodes[tmp_data[j]['id']].predict(tmp_data[j]['X'])
                    tmp_pred.append({'prob':prob.reshape(prob.shape[0], -1), 'idx':tmp_data[j]['idx']})
                    continue
                label = self.nodes[tmp_data[j]['id']].predict(tmp_data[j]['X'])
                for k in range(self.num_cluster):
                    idx = (label == k)
                    if idx.shape[0] == 0:
                        continue
                    tmp.append({'X':tmp_data[j]['X'][idx],
                                'idx':tmp_data[j]['idx'][idx],
                                'id':tmp_data[j]['id']+str(k)})
            tmp_data = tmp
        prob, idx = [], []
        for i in range(len(tmp_pred)):
            prob.append(tmp_pred[i]['prob'])
            idx.append(tmp_pred[i]['idx'])
        prob = np.concatenate(prob, axis=0)
        idx = np.concatenate(idx, axis=0)
        idx = np.argsort(idx)
        return prob[idx]
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, Y):
        return accuracy_score(Y, self.predict(X))


if __name__ == "__main__":
    
    from sklearn.svm import SVC
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    '''
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    print(" input feature shape: %s"%str(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2, stratify=digits.target)
    
    clf = HierKmeans(depth=3, learner=SVC(gamma='scale', probability=True), num_cluster=3, metric=None)
    clf.fit(X_train, y_train)
    print(clf.nodes.keys())
    print(" --> train acc: %s"%str(clf.score(X_train, y_train)))
    print(" --> test acc.: %s"%str(clf.score(X_test, y_test)))
    print("------- DONE -------\n")
    '''
    import cv2
    x = cv2.imread('train.jpg')
    y = cv2.imread('gt.jpg', 0)
    y = cv2.GaussianBlur(y, (5,5), 2)
    y = y.astype('float64')
    y /= 255
    y *= 5
    y = y.astype('int16')
    
    from cwSaab import cwSaab
    from skimage.util import view_as_windows

    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        X = np.pad(X, ((0,0),(win//2,win//2),(win//2,win//2),(0,0)), 'reflect')
        X = view_as_windows(X, (1,win,win,1), (1,1,1,1))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    def Concat(X, concatArg):
        X = np.concatenate(X, axis=-1)
        return X

    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None}, 
                {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None}]
    shrinkArgs = [{'func':Shrink, 'win':5}, 
                {'func': Shrink, 'win':5}]
    concatArg = {'func':Concat}
    inv_concatArg = {'func':Concat}

    X = x.reshape(1,321,481,3)
    cwsaab = cwSaab(depth=2, energyTH=0.01, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    output, DC = cwsaab.fit(X)
    X = output.reshape(-1, output.shape[-1])
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    
    clf = HierKmeans(depth=3, learner=RandomForestClassifier(n_jobs=-1), num_cluster=3, metric=None)
    clf.fit(X_train, y_train)
    print(" --> train acc: %s"%str(clf.score(X_train, y_train)))
    print(" --> test acc.: %s"%str(clf.score(X_test, y_test)))

