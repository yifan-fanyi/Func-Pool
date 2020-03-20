# 2020.03.19v1
# A generalized version of channel wise Saab
# Current code accepts <np.array> shape(..., D) as input
#
# Depth goal may not achieved is no nodes's energy is larger than energy threshold or too few SaabArgs/shrinkArgs, (warning generates)
#
import numpy as np 
import pickle

from saab import Saab

class cwSaab():
    def __init__(self, depth=1, energyTH=0.01, SaabArgs=None, shrinkArgs=None, concatArg=None):
        self.par = {}
        assert (depth > 0), "'depth' must > 0!"
        self.depth = (int)(depth)
        self.energyTH = energyTH
        assert (SaabArgs != None), "Need parameter 'SaabArgs'!"
        self.SaabArgs = SaabArgs
        assert (shrinkArgs != None), "Need parameter 'shrinkArgs'!"
        self.shrinkArgs = shrinkArgs
        assert (concatArg != None), "Need parameter 'concatArg'!"
        self.concatArg = concatArg
        self.Energy = []
        self.trained = False
        self.split = False
        if depth > np.min([len(SaabArgs), len(shrinkArgs)]):
            self.depth = np.min([len(SaabArgs), len(shrinkArgs)])
            print("       <WARNING> Too few 'SaabArgs/shrinkArgs' to get depth %s, actual depth: %s"%str(depth, self.depth))
        
    def SaabTransform(self, X, saab, train, layer):
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        X = shrinkArg['func'](X, shrinkArg)
        S = X.shape
        X = X.reshape(-1, S[-1])
        if train == True:
            saab = Saab(num_kernels=SaabArg['num_AC_kernels'], useDC=SaabArg['useDC'], needBias=SaabArg['needBias'])
            saab.fit(X)
        transformed = saab.transform(X)
        transformed = transformed.reshape(S)
        return saab, transformed

    def cwSaab_1_layer(self, X, train):
        S = list(X.shape)
        S[-1] = 1
        X = np.moveaxis(X, -1, 0)
        if train == True:
            saab_cur = []
        else:
            saab_cur = self.par['Layer'+str(0)]
        transformed, eng = [], []
        for i in range(X.shape[0]):
            X_tmp = X[i].reshape(S)
            if train == True:
                saab, tmp_transformed = self.SaabTransform(X_tmp, saab=None, train=True, layer=0)
                saab_cur.append(saab)
                eng.append(saab.Energy)
            else:
                if len(saab_cur) == i:
                    break
                _, tmp_transformed = self.SaabTransform(X_tmp, saab=saab_cur[i], train=False, layer=0)
            transformed.append(tmp_transformed)
        if train == True:
            self.par['Layer'+str(0)] = saab_cur
            self.Energy.append(np.concatenate(eng, axis=0))
        return np.concatenate(transformed, axis=-1)

    def cwSaab_n_layer(self, X, train, layer):
        output, eng_cur, ct, pidx = [], [], -1, 0
        S = list(X.shape)
        S[-1] = 1
        X = np.moveaxis(X, -1, 0)
        saab_prev = self.par['Layer'+str(layer-1)]
        if train == True:
            saab_cur = []
        else:
            saab_cur = self.par['Layer'+str(layer)]
        for i in range(len(saab_prev)):
            for j in range(saab_prev[i].Energy.shape[0]):
                ct += 1
                if saab_prev[i].Energy[j] < self.energyTH:
                    continue
                self.split = True
                X_tmp = X[ct].reshape(S)
                if train == True:
                    saab, out_tmp = self.SaabTransform(X_tmp, saab=None, train=True, layer=layer)
                    saab.Energy *= saab_prev[i].Energy[j]
                    saab_cur.append(saab)
                    eng_cur.append(saab.Energy) 
                else:
                    _, out_tmp = self.SaabTransform(X_tmp, saab=saab_cur[pidx], train=False, layer=layer)
                    pidx += 1
                output.append(out_tmp)
        if self.split == True:
            output = np.concatenate(output, axis=-1)
            if train == True:
                self.par['Layer'+str(layer)] = saab_cur
                self.Energy.append(np.concatenate(eng_cur, axis=0))
        return output
    
    def fit(self, X):
        output = []
        X = self.cwSaab_1_layer(X, train=True)
        output.append(X)
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train=True, layer=i)
            if self.split == False:
                self.depth = i
                print("       <WARNING> Cannot futher split, actual depth: %s"%str(i))
                break
            output.append(X)
        self.trained = True
        assert ('func' in concatArg.keys()), "'concatArg' must have key 'func'!"
        output = concatArg['func'](output, concatArg)
        self.Energy = np.concatenate(self.Energy, axis=0)
        return output

    def transform(self, X):
        assert (self.trained == True), "Must fit cwSaab first!"
        output = []
        X = self.cwSaab_1_layer(X, train=False)
        output.append(X)
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train=False, layer=i)
            output.append(X)
        assert ('func' in self.concatArg.keys()), "'concatArg' must have key 'func'!"
        output = self.concatArg['func'](output, concatArg)
        return output

if __name__ == "__main__":
    # example useage
    from sklearn import datasets
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
        X = np.concatenate(X, axis=-1)
        return X

    # read data

    print(" \n> This is a test enample: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(X.shape))

    # set args
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None},
                {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None}]
    shrinkArgs = [{'func':Shrink, 'dilate':[1], 'pad':'reflect'},
                {'func': Shrink, 'dilate':[1], 'pad':'reflect'}]
    concatArg = {'func':Concat}

    # run
    cwsaab = cwSaab(depth=2, energyTH=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    output = cwsaab.fit(X)
    print(" --> train feature shape: ", output.shape)
    output = cwsaab.transform(X)
    print(" --> test feature shape: ", output.shape)
    print("------- DONE -------\n")