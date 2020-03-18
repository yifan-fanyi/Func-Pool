# 2020.03.18
#
# regression on subspace
# particular designed for encounter missing class in this subspace
# if one class do not exists in training data, probability for this class would be zeros under anytime
#
#   learner, a regressor or classifier, must have methods named 'predict'
#   num_class: total number of class in dataset

import numpy as np 
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class myRegression():
    def __init__(self, learner, num_class):
        self.learner = learner
        self.num_class = num_class
        self.class_list = {}
    
    def mapping(self, Y, train=True):
        res = []
        Y = Y.reshape(-1)
        if train == True:
            c = 0
            for i in range(np.array(Y).shape[0]):
                if Y[i] not in self.class_list:
                    self.class_list[Y[i]] = c
                    c+=1
                res.append(self.class_list[Y[i]])
        else:
            for i in range(np.array(Y).shape[0]):
                for d in self.class_list.keys():
                    if self.class_list[d] == Y[i]:
                        res.append(d)
        return np.array(res)

    def fit(self, X, Y):
        Y = self.mapping(Y, train=True)
        self.learner.fit(X, Y)

    def predict(self, X):    
        tmp_pred = self.learner.predict(X).reshape(-1)
        pred= self.mapping(tmp_pred,  train=False)
        return pred

    def score(self, X, Y):
        return self.learner.score(X, Y) 

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    print(" \n> This is a test enample using MNIST: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.5, shuffle=False)
    clf = myRegression(LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr', n_jobs=20, max_iter=1000),
             10)
    clf.fit(X_train, y_train)
    print(" --> train acc: %s"%str(clf.score(X_train, y_train)))
    print(" --> test acc.: %s"%str(clf.score(X_test, y_test)))
    print("------- DONE -------\n")