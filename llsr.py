#v2020.02.15
import keras
import numpy as np
import sklearn
from numpy import linalg as LA

# solve cases with missing class
# ex, map [1,5,7] to [0,1,2]
def mapping(label, mydict=None, train=True):
    res = []
    label = label.reshape(-1)
    if train == True:
        c = 0
        for i in range(np.array(label).shape[0]):
            if label[i] not in mydict:
                mydict[label[i]] = c
                c+=1
            res.append(mydict[label[i]])
    else:
        for i in range(np.array(label).shape[0]):
            for d in mydict.keys():
                if mydict[d] == label[i]:
                    res.append(d)
    return np.array(res), mydict

def LLSR(X, Y=None, onehot=False, weight=None, train=True, probability=True):
    A = np.ones((X.shape[0], 1))
    X = np.concatenate((A, X), axis=1)
    if train == True:
        if onehot == False:
            y = keras.utils.to_categorical(Y, np.unique(Y).shape[0])
        weight = np.matmul(LA.pinv(X), y)
    pred = np.matmul(X, weight)
    if probability == False:
        pred = np.argmax(pred, axis=1)
    return pred, weight
