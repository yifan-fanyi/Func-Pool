# 2020.04.07
# A generalized version of channel wise Saab
# Current code accepts <np.array> shape(..., D) as input
#
# Depth goal may not achieved is no nodes's energy is larger than energy threshold or too few SaabArgs/shrinkArgs, (warning generates)
#
import numpy as np 
from sklearn.decomposition import PCA

from saab import Saab

class cwSaab():
    def __init__(self, depth=1, energyTH=0.01, SaabArgs=None, shrinkArgs=None, concatArg=None, splitMode=2, cwHop1=False):
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
        self.splitidx = []
        self.trained = False
        self.split = False
        self.splitMode = splitMode
        self.cwHop1 = cwHop1
        if depth > np.min([len(SaabArgs), len(shrinkArgs)]):
            self.depth = np.min([len(SaabArgs), len(shrinkArgs)])
            print("       <WARNING> Too few 'SaabArgs/shrinkArgs' to get depth %s, actual depth: %s"%(str(depth),str(self.depth)))

    def judge_abs_energy(self, eng):
        return (eng > self.energyTH)

    def judge_energy_ratio(self, X, R1, layer):
        X = self.shrinkArgs[layer]['func'](X, self.shrinkArgs[layer])
        X = X.reshape(-1, X.shape[-1]) - np.mean(X.reshape(-1, X.shape[-1]), axis=1, keepdims=True)
        pca = PCA(n_components=1, svd_solver='auto').fit(X)
        R2 = pca.explained_variance_ratio_[0]
        return (R1 / R2 >= self.energyTH)

    def judge_mean_abs_value(self, X, layer):
        X = self.shrinkArgs[layer]['func'](X, self.shrinkArgs[layer])
        tmp = np.moveaxis(X, -1, 0)[0]
        R1 = np.abs(tmp.reshape(-1, 1))
        R2 = np.mean(np.abs(X.reshape(-1, X.shape[-1])), axis=-1, keepdims=True)
        R = np.mean(R2 / R1)
        return (R > self.energyTH)

    def split_(self, X, eng, layer):
        if self.splitMode == 0:
            return self.judge_abs_energy(eng)
        elif self.splitMode == 1:
            return self.judge_mean_abs_value(X, layer)
        elif self.splitMode == 2:
            return self.judge_energy_ratio(X, eng, layer)
        else:
            raise ValueError("Unsupport split mode! Supported: 0, 1, 2")

    def SaabTransform(self, X, saab, train, layer):
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        X = shrinkArg['func'](X, shrinkArg)
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        if SaabArg['num_AC_kernels'] != -1:
            S[-1] = SaabArg['num_AC_kernels']
        if train == True:
            saab = Saab(num_kernels=SaabArg['num_AC_kernels'], useDC=SaabArg['useDC'], needBias=SaabArg['needBias'])
            saab.fit(X)
        transformed = saab.transform(X).reshape(S)
        return saab, transformed

    def cwSaab_1_layer(self, X, train):
        if train == True:
            saab_cur = []
        else:
            saab_cur = self.par['Layer'+str(0)]
        transformed, eng = [], []
        if train == True:
            saab, transformed = self.SaabTransform(X, saab=None, train=True, layer=0)
            saab_cur.append(saab)
            eng.append(saab.Energy)
        else:
            _, transformed = self.SaabTransform(X, saab=saab_cur[0], train=False, layer=0)
        if train == True:
            self.par['Layer'+str(0)] = saab_cur
            self.Energy.append(np.concatenate(eng, axis=0))
        return transformed

    def cwSaab_1_layer_cw(self, X, train):
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
            saab_cur, splitidx = [], []
        else:
            saab_cur = self.par['Layer'+str(layer)]
        for i in range(len(saab_prev)):
            for j in range(saab_prev[i].Energy.shape[0]):
                ct += 1
                X_tmp = X[ct].reshape(S)
                if train == True:
                    tidx = self.split_(X_tmp, saab_prev[i].Energy[j], layer)
                    splitidx.append(tidx)
                else:
                    tidx = self.splitidx[layer-1][ct]
                if tidx == False:
                    continue
                self.split = True
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
                self.splitidx.append(splitidx)
                self.par['Layer'+str(layer)] = saab_cur
                self.Energy.append(np.concatenate(eng_cur, axis=0))
        return output
    
    def fit(self, X):
        output = []
        if self.cwHop1 == False:
            X = self.cwSaab_1_layer(X, train=True)
        else:
            X = self.cwSaab_1_layer_cw(X, train=True)
        output.append(X)
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train=True, layer=i)
            if self.split == False:
                self.depth = i
                print("       <WARNING> Cannot futher split, actual depth: %s"%str(i))
                break
            output.append(X)
            self.split = False
        self.trained = True
        assert ('func' in self.concatArg.keys()), "'concatArg' must have key 'func'!"
        output = self.concatArg['func'](output, self.concatArg)
        return output

    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        output = []
        if self.cwHop1 == False:
            X = self.cwSaab_1_layer(X, train=False)
        else:
            X = self.cwSaab_1_layer_cw(X, train=False)
        output.append(X)
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train=False, layer=i)
            output.append(X)
        assert ('func' in self.concatArg.keys()), "'concatArg' must have key 'func'!"
        output = self.concatArg['func'](output, self.concatArg)
        return output
    
    def inv_SaabTransform(self, X, saab, inv_shrinkArg):
        assert ('func' in inv_shrinkArg.keys()), "'inv_shrinkArg' must contain key 'func'!"
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        X = saab.inverse_transform(X)
        S[-1] = np.array(X.shape)[-1]        
        X = X.reshape(S)
        X = inv_shrinkArg['func'](X, inv_shrinkArg)
        return X

    def inverse_transform(self, X, inv_concatArg, inv_shrinkArgs):
        assert (self.trained == True), "Must call fit first!"
        assert ('func' in inv_concatArg.keys()), "'inv_concatArg' must contain key 'func'!"
        X = inv_concatArg['func'](X, inv_concatArg)
        tmp = np.moveaxis(X[self.depth-1], -1, 0)
        for i in range(self.depth-1, -1, -1):
            res, ct = [], 0
            for j in range(len(self.par['Layer'+str(i)])):
                num_kernel = self.par['Layer'+str(i)][j].Energy.shape[0]
                res.append(self.inv_SaabTransform(np.moveaxis(tmp[ct:ct+num_kernel], 0, -1), 
                                                  saab=self.par['Layer'+str(i)][j],  
                                                  inv_shrinkArg=inv_shrinkArgs[i]))
                ct += num_kernel
            res = np.concatenate(res, axis=-1)  
            if i > 0:
                res = np.moveaxis(res, -1, 0)
                tmp = np.moveaxis(X[i-1], -1, 0)
                ct = 0
                for j in range(tmp.shape[0]):
                    if self.Energy[i-1][j] > self.energyTH:
                        tmp[j] = res[ct]
                        ct+=1
        return res
        
if __name__ == "__main__":
    # example useage
    from sklearn import datasets
    from skimage.util import view_as_windows

    # example callback function for collecting patches and its inverse
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

    # example callback function for how to concate features from different hops
    def Concat(X, concatArg):
        return X

    # read data
    import cv2
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(X.shape))

    # set args
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':False, 'batch':None}, 
                {'num_AC_kernels':-1, 'needBias':True, 'useDC':False, 'batch':None}]
    shrinkArgs = [{'func':Shrink, 'win':2}, 
                {'func': Shrink, 'win':2}]
    inv_shrinkArgs = [{'func':invShrink, 'win':2}, 
                    {'func': invShrink, 'win':2}]
    concatArg = {'func':Concat}
    inv_concatArg = {'func':Concat}

    print(" --> test inv")
    print(" -----> depth=1")
    cwsaab = cwSaab(depth=1, energyTH=0.1, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    output = cwsaab.fit(X)
    output = cwsaab.transform(X)
    Y = cwsaab.inverse_transform(output, inv_concatArg=inv_concatArg, inv_shrinkArgs=inv_shrinkArgs)
    Y = np.round(Y)
    assert (np.mean(np.abs(X-Y)) < 1e-5), "invcwSaab error!"
    print(" -----> depth=2")
    cwsaab = cwSaab(depth=2, energyTH=0.5, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg, splitMode=2, cwHop1=True)
    output = cwsaab.fit(X)
    output = cwsaab.transform(X)
    Y = cwsaab.inverse_transform(output, inv_concatArg=inv_concatArg, inv_shrinkArgs=inv_shrinkArgs)
    Y = np.round(Y)
    assert (np.mean(np.abs(X-Y)) < 1), "invcwSaab error!"
    print(output[0].shape, output[1].shape)
    print("------- DONE -------\n")