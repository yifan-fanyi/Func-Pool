# 2020.04.06

import numpy as np 
import pickle
import cv2
import os
from skimage.util import view_as_windows

def Shrink(X, shrinkArg):
    win = shrinkArg['win']
    X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def invShrink(X, invshrinkArg):
    win = invshrinkArg['win']
    S = X.shape
    X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
    X = np.moveaxis(X, 5, 2)
    X = np.moveaxis(X, 6, 4)
    return X.reshape(S[0], win*S[1], win*S[2], -1)

def Concat(X, concatArg):
    return X

def load_by_channel(path, channel=0, count=10, size=None):
    name = os.listdir(path)
    raw, ct = [], 0
    for n in name:
        x = cv2.imread(path+n)
        try:
            x.shape
        except:
            continue
        x = cv2.cvtColor(x, cv2.COLOR_BGR2YCR_CB)[:,:,channel]
        if x.shape[0] > x.shape[1]:
            x = np.transpose(x)
        x = x.reshape(x.shape[0], x.shape[1], 1)
        raw.append(x)
        ct+=1
        if ct == count:
            break
    return np.array(raw)

def load_RGB(path, count=10, size=None):
    name = os.listdir(path)
    raw, ct = [], 0
    for n in name:
        x = cv2.imread(path+n)
        try:
            x.shape
        except:
            continue
        if x.shape[0] > x.shape[1]:
            r = np.transpose(x[:,:,0]).reshape(x.shape[1],x.shape[0],1)
            g = np.transpose(x[:,:,1]).reshape(x.shape[1],x.shape[0],1)
            b = np.transpose(x[:,:,2]).reshape(x.shape[1],x.shape[0],1)
            x = np.concatenate((r,g,b), axis=-1)
        raw.append(x)
        ct+=1
        if ct == count:
            break
    return np.array(raw)

def load_YUV(path, count=10, size=None):
    name = os.listdir(path)
    raw, ct = [], 0
    for n in name:
        x = cv2.imread(path+n)
        try:
            x.shape
        except:
            continue
        x = cv2.cvtColor(x, cv2.COLOR_BGR2YCR_CB)
        if x.shape[0] > x.shape[1]:
            r = np.transpose(x[:,:,0]).reshape(x.shape[1],x.shape[0],1)
            g = np.transpose(x[:,:,1]).reshape(x.shape[1],x.shape[0],1)
            b = np.transpose(x[:,:,2]).reshape(x.shape[1],x.shape[0],1)
            x = np.concatenate((r,g,b), axis=-1)
        raw.append(x)
        ct+=1
        if ct == count:
            break
    return np.array(raw)
class calQ_Factor():
    def __init__(self, Lambda=None, Sigma=None):
        assert (Lambda != None or Sigma != None), "Lambda or Sigma should be int!"            
        if Sigma != None and Lambda != None:
            tmp = 1.4142135623731 / Sigma
            if abs(tmp - Lambda) > 1e-5:
                print("   <Warning> Lambda and Sigma not match, use Lambda!")
            tmp = Lambda
        elif Sigma != None:
            tmp = 1.4142135623731 / Sigma
        else:
            tmp = Lambda
        self.Lambda = tmp
            
    def laplacian_(self, k, Q, x):
        return 2 * k * Q * x + \
               2 * k * Q / self.Lambda - \
               (k**2) * (Q**2) - \
               x * x - \
               x / (self.Lambda**2) - \
               1 / self.Lambda
    
    def integral_(self, Q):
        res = 0.
        for k in range(1, 10000):
            res += np.exp(self.Lambda * k*Q+Q/2) * self.laplacian_(k, Q, k*Q+Q/2) - \
                   np.exp(self.Lambda * k*Q+Q/2) * self.laplacian_(k, Q, k*Q-Q/2)
        return res
    
    def cal(self):
        Q = np.arange(0, 1000)
        val = self.integral_(Q)
        return Q[np.argmin(val)]
 