# 2021.05.12
# @yifan
#
# XGBoost Classifier
#
import numpy as np
import xgboost as xgb

class XGBC():
    def __init__(self, n_iter=20, num_class=2, max_depth=3, param={}):
        self.n_iter = n_iter
        self.param = { 'eta': 0.3, 
                       'max_depth': max_depth,  
                       'objective': 'multi:softprob',  
                       'num_class': 3} 
        self.param.update(param)
        self.learner = None
        
    def fit(self, X, Y):
        data = xgb.DMatrix(X, label=Y)
        self.learner = xgb.train(self.param, data, self.n_iter)
        return self
    
    def predict_proba(self, X):
        data = xgb.DMatrix(X, label=None)
        prob = self.learner.predict(data)
        return prob
    
    def predict(self, X):
        prob = self.predict_proba(X)
        pred = np.asarray([np.argmax(line) for line in prob])
        return pred
if __name__ == "__main__":   
    from sklearn import datasets
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    xbgc = XGBC()
    xbgc.fit(X, y)
    pred = xbgc.predict(X)
    Y_test = y
    best_preds = pred
    print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))
    print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))
    print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))