#v2020.03.18
import keras
import numpy as np
import sklearn
from numpy import linalg as LA

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
