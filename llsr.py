#v2020.03.19
import numpy as np
import scipy
from sklearn.metrics import accuracy_score

class LLSR():
    def __init__(self, weight=None, bias=None, onehot=False):
        self.weight = weight
        self.bias = bias
        self.onehot = False

    def fit(self, X, Y):
        if self.onehot == True:
            y = np.zeros((X.shape[0], np.unique(Y).shape[0]))
            y[np.arange(Y.size), Y] = 1
            Y = y.copy()
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((A, X), axis=1)
        weight = scipy.linalg.lstsq(X, Y)[0]       
        self.weight = weight[1:weight.shape[0]]
        self.bias = weight[0].reshape(1, -1)

    def predict(self, X):
        X = self.predict_proba(X)
        return np.argmax(X, axis=1)

    def predict_proba(self, X):
        X = np.matmul(X, self.weight)
        return X + self.bias

    def score(self, X, Y):
        pred = self.predict(X)
        return accuracy_score(Y, pred)
    
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    print(" \n> This is a test enample: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    print(" input feature shape: %s"%str(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2,  stratify=digits.target)
    
    clf = LLSR()
    clf.fit(X_train, y_train)
    print(" --> train acc: %s"%str(clf.score(X_train, y_train)))
    print(" --> test acc: %s"%str(clf.score(X_test, y_test)))
    print("------- DONE -------\n")
