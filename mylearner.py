# 2020.05.10
#  update topNscore
# learner on subspace
# particular designed for encounter missing class in this subspace
# if one class do not exists in training data, probability for this class would be zeros under anytime
#
#   learner: a regressor or classifier, must have methods named 'predict'
#   num_class: total number of class in dataset

import numpy as np 
from sklearn.metrics import accuracy_score

class myLearner():
    def __init__(self, learner, num_class):
        self.learner = learner
        self.num_class = num_class
        self.class_list = {}
        self.oneclass = False
        self.trained = False
    
    def mapping(self, Y, train=True, probability=False):
        c, res = 0, []
        Y = Y.reshape(Y.shape[0], -1)
        if train == True:
            self.class_list = {}
            for i in range(np.array(Y).shape[0]):
                if Y[i, 0] not in self.class_list.keys():
                    self.class_list[Y[i,0]] = c
                    c += 1
                res.append(self.class_list[Y[i, 0]])
        else:
            if probability == False:
                for i in range(np.array(Y).shape[0]):
                    for d in self.class_list.keys():
                        if self.class_list[d] == Y[i, 0]:
                            res.append(d)
            else:
                res = np.zeros((Y.shape[0], self.num_class))
                for i in range(np.array(Y).shape[0]):
                    c = 0
                    for j in range(self.num_class):
                        if j in self.class_list.keys():
                            res[i, j] = Y[i, self.class_list[j]]
                            c += 1
        return np.array(res)

    def fit(self, X, Y):
        Y = self.mapping(Y, train=True)
        if np.unique(Y).shape[0] == 1:
            self.oneclass = True
        else:
            self.learner.fit(X, Y)
        self.trained = True
        return self

    def predict(self, X): 
        assert (self.trained == True), "Must call fit first!"
        if self.oneclass == False:
            tmp_pred = self.learner.predict(X).reshape(-1)
        else:
            tmp_pred = np.zeros((X.shape[0]))
        return self.mapping(tmp_pred, train=False)

    def predict_proba(self, X): 
        assert (self.trained == True), "Must call fit first!"
        if self.oneclass == False:
            tmp_pred = self.learner.predict_proba(X)
        else:
            tmp_pred = np.ones((X.shape[0], 1))
        return self.mapping(tmp_pred, train=False, probability=True)

    def score(self, X, Y):
        assert (self.trained == True), "Must call fit first!"
        return accuracy_score(Y, self.predict(X)) 

    def topNscore(self, X, Y, N=3):
        prob = self.predict_proba(X)
        idx = np.argsort(prob, axis=1)
        ct = 0.
        Y = Y.astype('int16')
        for i in range(len(Y)):
            if Y[i] in (list)(idx[i, -N:]):
                ct+=1
        return ct/(float)(len(Y))

if __name__ == "__main__":
    from sklearn.svm import SVC
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    print(" input feature shape: %s"%str(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2, stratify=digits.target)
    
    clf = myLearner(SVC(gamma='scale', probability=True), 10)
    clf.fit(X_train, y_train)
    print(" --> train acc: %s"%str(clf.score(X_train, y_train)))
    print(" --> test acc.: %s"%str(clf.score(X_test, y_test)))
    print(" --> test top3 acc.: %s"%str(clf.topNscore(X_test, y_test, 3)))
    print("------- DONE -------\n")
