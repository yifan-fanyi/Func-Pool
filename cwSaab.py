# v2020.02.15
# A generalized version of channel wise Saab
# Current code accepts <np.array> shape(..., D) as input
#
# Depth goal may not achieved is no nodes's energy is larger than energy threshold, (warning generates)
#
# train = True:
#   'depth', 'energyTH', 'SaabArgs', 'shrinkArgs', 'concatArgs' are needed, (len(SaabArgs) and len(shrinkArgs) should equal or larger than 'depth')
#   will train Saab transformation and save the parameters
# train = False:
#   'par' is needed, others args is unnecessay
#   other setting will be loaded from 'par'
#
# detail usage can be found in the example start from line170

import numpy as np 
import pickle

from saab import Saab

def Transform(X, par, train, shrinkArg, SaabArg):
    '''
    Collecting patches and doing Saab transformation
        X: <np.array> feature to be transformed
        par: <dict> Saab parameters to transform X when train=False, else not used
        train: <bool> indicate whether is training or testing
        shrinkArg: <dict> passed to func 'func:Shrink'
        SaabArg: <dict> passed to 'func:Saab'
    
    return: <dict> Saab parameters, 
            <np.array> transformed feature, only last dimension may change
    '''
    X = shrinkArg['func'](X, shrinkArg)
    S = X.shape
    X = X.reshape(-1, S[-1])
    transformed, par = Saab(None, num_kernels=SaabArg['num_AC_kernels'], useDC=SaabArg['useDC'], batch=SaabArg['batch'], needBias=SaabArg['needBias']).Saab_transform(X, train=train, pca_params=par)
    transformed = transformed.reshape(S)
    return par, transformed

def cwSaab_1_layer(X, train, par_cur, SaabArg, shrinkArg):
    '''
    First layer of cwSaab
        X: <np.array> 
        train: <bool> indicate whether is training or testing
        par_cur: <list of dict> Saab parameters to transform X when train=False, else not used
        shrinkArg: <dict>
        SaabArg: <dict>
    
    return: <np.array> transformed feature, only last dimension may change, 
            <list of dict> Saab parameters 
            <np.array> shape(D) energy for each kernel
    '''
    S = list(X.shape)
    S[-1] = 1
    X = np.moveaxis(X, -1, 0)
    if train == True:
        par_cur = []
    transformed, eng = [], []
    for i in range(X.shape[0]):
        X_tmp = X[i].reshape(S)
        if train == True:
            par, tmp_transformed = Transform(X_tmp, par=None, train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
            par_cur.append(par)
        else:
            if len(par_cur) == i:
                break
            par, tmp_transformed = Transform(X_tmp, par=par_cur[i], train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
        transformed.append(tmp_transformed)
        eng.append(par['Energy'])
    return np.concatenate(transformed, axis=-1), par_cur, np.concatenate(eng, axis=0)

def cwSaab_n_layer(X, energyTH, train, par_prev, par_cur, SaabArg, shrinkArg):
    '''
    n^th layer of cwSaab
        X: <np.array> shape(..., D)
        energyTH: <float> energy threshold
        train: <bool> indicate whether is training or testing
        par_prev: <list of dict> Saab parameter of previous hop
        par_cur: <list of dict> Saab parameters to transform X when train=False, else not used
        shrinkArg: <dict>
        SaabArg: <dict>

    return: <np.array> transformed feature, only last dimension may change, 
            <list of dict> Saab parameters 
            <np.array> shape(D) energy for each kernel
            <bool>: whether this hop contain new nodes
    '''
    output, eng_cur = [], []
    S = list(X.shape)
    S[-1] = 1
    X = np.moveaxis(X, -1, 0)
    ct, split = -1, False
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
            split = True
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
    if split == True:
        output = np.concatenate(output, axis=-1)
        eng_cur = np.concatenate(eng_cur, axis=0)
    return output, par_cur, eng_cur, split
    
def cwSaab(X, train, par=None, depth=None, energyTH=None, SaabArgs=None, shrinkArgs=None, concatArg=None):
    '''
    main function
        X: <np.array> shape(..., D)
        train: <bool> indicate whether is training or testing
        par: <dict> parameter used when train=False
        depth: <int> depth of tree
        energyTH: <float> energy threshold
        shrinkArgs: <list of dict> callback function and its arguments for how to collecting patches in each hop, 
                    each element must have key 'func' (function name of outside method)
        SaabArg: <list of dict> arguments for each hop
        concatArg: <dict> callback function and its arguments for how to concat features
                    must have key 'func' (function name of outside method)

    return: <any> depends on return of concatArg['func']
            <dict> Saab parameters 
    '''
    output, eng = [], []
    if train == True:
        par = {'depth': depth, 'energyTH': energyTH, 'SaabArgs': SaabArgs, 'shrinkArgs': shrinkArgs, 'concatArg': concatArg}
        X, par_tmp, eng_tmp= cwSaab_1_layer(X, train=train, par_cur=[], SaabArg=SaabArgs[0], shrinkArg=shrinkArgs[0])
        output.append(X)
        eng.append(eng_tmp)
        par['Layer0'] = par_tmp
        for i in range(1, depth):
            X, par_tmp, eng_tmp, split = cwSaab_n_layer(X, energyTH=energyTH, train=train, par_prev=par_tmp, par_cur=[], SaabArg=SaabArgs[i], shrinkArg=shrinkArgs[i])
            if split == False:
                par['depth'], depth = i, i
                print("       <WARNING> Cannot futher split, actual depth: %s"%str(i))
                break
            output.append(X)
            eng.append(eng_tmp)
            par['Layer'+str(i)] = par_tmp
    else:
        depth, energyTH, shrinkArgs, SaabArgs, concatArg = par['depth'], par['energyTH'], par['shrinkArgs'], par['SaabArgs'], par['concatArg']
        X, par_tmp, eng_tmp= cwSaab_1_layer(X, train=train, par_cur=par['Layer0'], SaabArg=SaabArgs[0], shrinkArg=shrinkArgs[0])
        output.append(X)
        eng.append(eng_tmp)
        for i in range(1, depth):
            X, par_tmp, eng_tmp, split = cwSaab_n_layer(X, energyTH=energyTH, train=train, par_prev=par['Layer'+str(i-1)], par_cur=par['Layer'+str(i)], SaabArg=SaabArgs[i], shrinkArg=shrinkArgs[i])
            output.append(X)
            eng.append(eng_tmp)
    for i in range(depth-1):
        output[i] = np.moveaxis(output[i], -1, 0)
        output[i] = output[i][eng[i] < energyTH]
        output[i] = np.moveaxis(output[i], 0, -1)
    output = concatArg['func'](output, concatArg)
    eng = np.concatenate(eng, axis=0)
    return output, par

if __name__ == "__main__":
    # example useage
    import cv2
    from pixelhop import PixelHop_Neighbour

    # example callback function for collecting patches
    def Shrink(X, shrinkArg):
        # only can have following two args
        #   X: <np.array> , data/feature generated inside the tree 
        #   shrinkArg: <dict> arguments needed to call outside methods
        #
        # return <np.array> same structure as data flow in tree
        return PixelHop_Neighbour(X, shrinkArg['dilate'], shrinkArg['pad'])

    # example callback function for how to concate features from different hops
    def Concat(X, concatArg):
        # only can have following two args
        #   X: <list> , feature of different hops
        #   concatArg: <dict> arguments needed to call outside methods
        #
        # return <any> it would become the output of tree
        return np.concatenate(X, axis=-1)

    # read data
    X = [cv2.imread('./data/test.jpg')]#, cv2.imread('./data/se.jpg')]
    X = np.array(X)
    s = [-1, 321, 481, 3]
    X = X.reshape(s)
    print("Input shape: ", X.shape)

    # set args
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None},
                {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None}]
    shrinkArgs = [{'func':Shrink, 'dilate':[1], 'pad':'reflect'},
                {'func': Shrink, 'dilate':[2], 'pad':'reflect'}]
    concatArg = {'func':Concat}

    # run
    output, par = cwSaab(X, train=True, par=None, depth=2, energyTH=0.001,  SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    print("train feature shape: ", output.shape)
    output, par = cwSaab(X, train=False, par=par)
    print("test feature shape: ", output.shape)