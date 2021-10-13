# 2021.10.13
# @yifan 
# Shrink, invShrink support rect window
#
import numpy as np
from skimage.util import view_as_windows
from LANCZOS import LANCZOS

def Shrink(X, win):
    if type(win) == dict:
        win = win['win']
    try:
        X = view_as_windows(X, (1,win[0],win[1],1), (1,win[0],win[1],1))
    except:
        X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def invShrink(X, win):
    if type(win) == dict:
        win = win['win']
    S = X.shape
    try:
        X = X.reshape(S[0], S[1], S[2], -1, 1, win[0], win[1], 1)
    except: 
        X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
    X = np.moveaxis(X, 5, 2)
    X = np.moveaxis(X, 6, 4)
    try:
        X = X.reshape(S[0], win[0]*S[1], win[1]*S[2], -1)
    except:
        X = X.reshape(S[0], win*S[1], win*S[2], -1)
    return X

def DownSample(X, r):
    a, b =  LANCZOS.split(X, r)   
    return a