# v2020.02.05
# a generialzed version of channel wise PCA
#
# Shrink: to support different type of data (image, pointcloud)
# Output_Concat: how to concatenate features from different hop, especially when spatial shape is different
# 
# train: <bool> 
# par: <dict>, parameters
# depth: <int>, depth of tree
# energtTH: <float>, energy threshold for stopping spliting on nodes
# SaabArgs: <list>, ex: [{'needBias':False, 'useDC':True, 'batch':None}]
# shrinkArgs: <list>, ex: [{'dilate':[1], 'pad':'reflect'}]
# concatArgs: <list>, currently not used, left for future
#
# during testing, settings like depth, SaabArgs, shrinkArgs will be loaded from par

import numpy as np 
import pickle

from saab import Saab
from pixelhop import PixelHop_Neighbour

def Shrink(X, shrinkArg):
    X = PixelHop_Neighbour(X, shrinkArg['dilate'], shrinkArg['pad'])
    return X

def Output_Concat(X, concatArgs):
    output = np.concatenate(X, axis=-1)
    return output

def Transform(X, par, train, shrinkArg, SaabArg):
    X = Shrink(X, shrinkArg=shrinkArg)
    S = X.shape
    X = X.reshape(-1, S[-1])
    transformed, par = Saab(None, S[-1], useDC=SaabArg['useDC'], batch=SaabArg['batch'], needBias=SaabArg['needBias']).Saab_transform(X, train=train, pca_params=par)
    transformed = transformed.reshape(S)
    return par, transformed

def cwPCA_1_layer(X, train, par_cur, SaabArg, shrinkArg):
    par, transformed = Transform(X, par=par_cur, train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
    return transformed, [par], par['Energy']

def cwPCA_n_layer(X, energyTH, train, par_prev, par_cur, SaabArg, shrinkArg):
    output = []
    eng_cur = []
    S = list(X.shape)
    S[-1] = 1
    X = np.moveaxis(X, -1, 0)
    ct = -1
    if train == True:
        par_cur = []
    else:
        pidx = 0
    for i in range(len(par_prev)):
        for j in range(par_prev[i]['Energy'].shape[0]):
            ct += 1
            if par_prev[i]['Energy'][j] < energyTH:
                continue
            X_tmp = X[ct].reshape(S)
            if train == True:
                par_tmp, out_tmp = Transform(X_tmp, par=None, train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
                par_tmp['Energy'] *= par_prev[i]['Energy'][j]
                eng_cur.append(par_tmp['Energy'])
                par_cur.append(par_tmp)
                output.append(out_tmp)
            else:
                par_tmp, out_tmp = Transform(X_tmp, par=par_cur[pidx], train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
                output.append(out_tmp)
                eng_cur.append(par_cur[pidx]['Energy'])
                pidx += 1
    output = np.concatenate(output, axis=-1)
    eng_cur = np.concatenate(eng_cur, axis=0)
    return output, par_cur, eng_cur

def cwPCA(X, train=True, par=None, depth=None, energyTH=None, SaabArgs=None, shrinkArgs=None, concatArgs=None):
    output = []
    eng = []
    if train == True:
        par = {'depth': depth, 'energyTH': energyTH, 'SaabArgs': SaabArgs, 'shrinkArgs': shrinkArgs, 'concatArgs': concatArgs}
        X, par_tmp, eng_tmp= cwPCA_1_layer(X, train=train, par_cur=[], SaabArg=SaabArgs[0], shrinkArg=shrinkArgs[0])
        output.append(X)
        eng.append(eng_tmp)
        par['Layer0'] = par_tmp
        for i in range(1, depth):
            X, par_tmp, eng_tmp = cwPCA_n_layer(X, energyTH=energyTH, train=train, par_prev=par_tmp, par_cur=[], SaabArg=SaabArgs[i], shrinkArg=shrinkArgs[i])
            output.append(X)
            eng.append(eng_tmp)
            par['Layer'+str(i)] = par_tmp
    else:
        depth = par['depth']
        energyTH = par['energyTH']
        shrinkArgs = par['shrinkArgs']
        SaabArgs = par['SaabArgs']
        concatArgs = par['concatArgs']
        X, par_tmp, eng_tmp= cwPCA_1_layer(X, train=train, par_cur=par['Layer0'][0], SaabArg=SaabArgs[0], shrinkArg=shrinkArgs[0])
        output.append(X)
        eng.append(eng_tmp)
        for i in range(1, depth):
            X, par_tmp, eng_tmp = cwPCA_n_layer(X, energyTH=energyTH, train=train, par_prev=par['Layer'+str(i-1)], par_cur=par['Layer'+str(i)], SaabArg=SaabArgs[i], shrinkArg=shrinkArgs[i])
            output.append(X)
            eng.append(eng_tmp)
    for i in range(depth-1):
        output[i] = np.moveaxis(output[i], -1, 0)
        output[i] =output[i][eng[i] < energyTH]
        output[i] = np.moveaxis(output[i], 0, -1)
    output = Output_Concat(output, concatArgs=concatArgs)
    eng = np.concatenate(eng, axis=0)
    return output, par

if __name__ == "__main__":
    import cv2
    X = cv2.imread('./data/test.jpg')
    s = [1, 321, 481, -1]
    X = X.reshape(s)
    print("Input shape: ", X.shape)
    SaabArgs = [{'needBias':False, 'useDC':True, 'batch':None},
                {'needBias':True, 'useDC':True, 'batch':None},
                {'needBias':True, 'useDC':True, 'batch':None}]
    shrinkArgs = [{'dilate':[1], 'pad':'reflect'},
                {'dilate':[2], 'pad':'reflect'},
                {'dilate':[4], 'pad':'reflect'}]
    output, par = cwPCA(X, train=True, par=None, depth=3, energyTH=0.001,  SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArgs=None)
    print("train feature shape: ", output.shape)
    output, par = cwPCA(X, train=False, par=par)
    print("test feature shape: ", output.shape)