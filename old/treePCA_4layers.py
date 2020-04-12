# v2020.01.18
# special case
# a cwPCA tree with depth 4

import numpy as np
import sklearn
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from numpy import linalg as LA

# copy from pixelhop.py v2020.01.18
def PixelHop_Neighbour(feature, dilate, pad):
    dilate = np.array(dilate)
    idx = [1, 0, -1]
    H, W = feature.shape[1], feature.shape[2]
    res = feature.copy()
    if pad == 'reflect':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'constant', constant_values=0)
    else:
        H, W = H - 2*dilate[-1], W - 2*dilate[-1]
        res = feature[:, dilate[-1]:dilate[-1]+H, dilate[-1]:dilate[-1]+W].copy()
    for d in range(dilate.shape[0]):
        for i in idx:
            for j in idx:
                if i == 0 and j == 0:
                    continue
                else:
                    ii, jj = (i+1)*dilate[d], (j+1)*dilate[d]
                    res = np.concatenate((feature[:, ii:ii+H, jj:jj+W], res), axis=3)
    return res 

def remove_mean(feature, axis):
    feature_mean = np.mean(feature, axis=axis, keepdims=True)
    feature = feature - feature_mean
    return feature, feature_mean

def Saab(X, train=1, num_kernels=1, params=None, Bias=True):
    par = {}
    if train == 1:
        XX, no = remove_mean(X.copy(), axis=0)
        XX, dc = remove_mean(X.copy(), axis=1)
        pca = PCA(n_components=num_kernels, svd_solver='full').fit(XX)
        kernels = pca.components_
        largest_ev = np.var(dc*np.sqrt(X.shape[-1]))
        dc_kernel = 1/np.sqrt(X.shape[-1])*np.ones((1, X.shape[-1]))/np.sqrt(largest_ev)
        kernels = np.concatenate((dc_kernel, kernels[:-1]), axis=0)
        bias = LA.norm(X, axis=1)
        bias = np.max(bias)
        if Bias is True:
            X += bias
        transformed = np.matmul(X, np.transpose(kernels))
        if Bias is True:
            e = np.zeros((1, kernels.shape[0]))
            e[0, 0] = 1
            transformed -= bias*e
        energy = np.concatenate((np.array([largest_ev]),pca.explained_variance_[:-1]), axis=0)
        energy = energy/np.sum(energy)
        par = {'Kernels':kernels, 'Bias':bias, 'Energy':energy, 'isLeaf':np.zeros((transformed.shape[1]))}
    else:
        kernels = params['Kernels']
        bias = params['Bias']
        if Bias is True:
            X += bias
        transformed = np.matmul(X, np.transpose(kernels))
        if Bias is True:
            e = np.zeros((1, kernels.shape[0]))
            e[0, 0] = 1
            transformed -= bias * e
    return transformed, par

def tree(X, params=None, train=1, bias=True, dilate=1):
    S = X.shape
    X = PixelHop_Neighbour(X, [dilate], pad='reflect')
    X = X.reshape(-1, X.shape[-1])
    if train == 1:
        transformed, params = Saab(X, train=1, num_kernels=X.shape[-1], params=None, Bias=bias)
    else:
        transformed , no = Saab(X, train=0, num_kernels=X.shape[-1], params=params, Bias=bias)
    transformed = transformed.reshape(S[0], S[1], S[2], -1)
    return params, transformed

def treePCA_train(X, energy_threshold=0.001, dilate = [1,2,3,4]):
    pca_params = {}
    feature_train = {}
    feature_output = []
    for i in range(4):
        if i == 0:
            params, output = tree(X, params=None, train=1, bias=0, dilate=dilate[i])
            params['isLeaf'][params['Energy']<energy_threshold] += 1
            pca_params['Layer_{:d}'.format(i)] = params
            feature_train['Layer_{:d}'.format(i)] = output
            feature_output.append(output[:, :, :, params['isLeaf'] > 0.5])
            print(pca_params.keys())
        elif i == 1:
            par = []
            feature = feature_train['Layer_{:d}'.format(i-1)]
            par = pca_params['Layer_{:d}'.format(i-1)]
            l1 = par['isLeaf'].shape[0]- (int)(np.sum(par['isLeaf']))
            for j in range(l1):
                if par['Energy'][j] < energy_threshold:
                    continue
                fea_tmp = feature[:, :, :, j].reshape(feature.shape[0], feature.shape[1], feature.shape[2], 1)
                params, output = tree(fea_tmp, params=None, train=1, bias=1, dilate=dilate[i])
                params['Energy'] *= par['Energy'][j]
                params['isLeaf'][params['Energy']<energy_threshold] += 1
                pca_params['Layer_{:d}_{:d}'.format(i, j)] = params
                feature_train['Layer_{:d}_{:d}'.format(i, j)] = output
                feature_output.append(output[:, :, :, params['isLeaf'] > 0.5])
            print(pca_params.keys())
        elif i == 2:
            for j in range(l1):
                if 'Layer_{:d}_{:d}'.format(i-1, j) not in pca_params:
                    continue
                feature = feature_train['Layer_{:d}_{:d}'.format(i-1, j)]
                par = pca_params['Layer_{:d}_{:d}'.format(1, j)]
                l2 = par['isLeaf'].shape[0]- (int)(np.sum(par['isLeaf']))
                for k in range(l2):
                    if par['Energy'][k] < energy_threshold:
                        continue
                    fea_tmp = feature[:, :, :, k].reshape(feature.shape[0], feature.shape[1], feature.shape[2], 1)
                    params, output = tree(fea_tmp, params=None, train=1, bias=1, dilate=dilate[i])
                    params['Energy'] *= par['Energy'][k]
                    params['isLeaf'][params['Energy']<energy_threshold] += 1
                    pca_params['Layer_{:d}_{:d}_{:d}'.format(i, j, k)] = params
                    feature_train['Layer_{:d}_{:d}_{:d}'.format(i, j, k)] = output
                    feature_output.append(output[:, :, :, params['isLeaf'] > 0.5])  
            print(pca_params.keys()) 
        elif i == 3:
            for j in range(l1):
                par = pca_params['Layer_{:d}_{:d}'.format(1, j)]
                l2 = par['isLeaf'].shape[0]- (int)(np.sum(par['isLeaf']))
                for k in range(l2):
                    if 'Layer_{:d}_{:d}_{:d}'.format(i-1, j, k) not in pca_params:
                        continue
                    feature = feature_train['Layer_{:d}_{:d}_{:d}'.format(i-1, j, k)]
                    par = pca_params['Layer_{:d}_{:d}_{:d}'.format(2, j, k)]
                    l3 = par['isLeaf'].shape[0]- (int)(np.sum(par['isLeaf']))
                    for t in range(l3):
                        if par['Energy'][t] < energy_threshold:
                            continue
                        fea_tmp = feature[:, :, :, t].reshape(feature.shape[0], feature.shape[1], feature.shape[2], 1)
                        params, output = tree(fea_tmp, params=None, train=1, bias=1, dilate=dilate[i])
                        params['Energy'] *= par['Energy'][t]
                        params['isLeaf'] += 1 
                        pca_params['Layer_{:d}_{:d}_{:d}_{:d}'.format(i, j, k, t)] = params
                        feature_train['Layer_{:d}_{:d}_{:d}_{:d}'.format(i, j, k, t)] = output
                        feature_output.append(output[:, :, :, params['isLeaf'] > 0.5])
            print(pca_params.keys())
    return np.concatenate(feature_output, axis=3), pca_params

def treePCA_test(X, pca_params, dilate = [1,2,3,4]):
    feature_test = {}
    feature_output = []
    for i in range(4):
        if i == 0:
            params = pca_params['Layer_{:d}'.format(i)]
            no, output = tree(X, params=params, train=0, bias=0, dilate=dilate[i])
            feature_test['Layer_{:d}'.format(i)] = output
            feature_output.append(output[:, :, :, params['isLeaf'] > 0.5])
        elif i == 1:
            feature = feature_test['Layer_{:d}'.format(i - 1)]
            par = pca_params['Layer_{:d}'.format(i-1)]
            l1 = par['isLeaf'].shape[0]- (int)(np.sum(par['isLeaf']))
            for j in range(l1):
                params = pca_params['Layer_{:d}_{:d}'.format(i, j)]
                fea_tmp = feature[:, :, :, j].reshape(feature.shape[0], feature.shape[1], feature.shape[2], 1)
                no, output = tree(fea_tmp, params=params, train=0, bias=1, dilate=dilate[i])
                feature_test['Layer_{:d}_{:d}'.format(i, j)] = output
                feature_output.append(output[:, :, :, params['isLeaf'] > 0.5])
        elif i == 2:
            for j in range(l1):
                if 'Layer_{:d}_{:d}'.format(i - 1, j) not in pca_params:
                    continue
                feature = feature_test['Layer_{:d}_{:d}'.format(i - 1, j)]
                par = pca_params['Layer_{:d}_{:d}'.format(i - 1, j)]
                l2 = par['isLeaf'].shape[0]- (int)(np.sum(par['isLeaf']))
                for k in range(l2):
                    if 'Layer_{:d}_{:d}_{:d}'.format(i, j, k) not in pca_params:
                        continue
                    params = pca_params['Layer_{:d}_{:d}_{:d}'.format(i, j, k)]
                    fea_tmp = feature[:, :, :, k].reshape(feature.shape[0], feature.shape[1], feature.shape[2], 1)
                    no, output = tree(fea_tmp, params=params, train=0, bias=1, dilate=dilate[i])
                    feature_test['Layer_{:d}_{:d}_{:d}'.format(i, j, k)] = output
                    feature_output.append(output[:, :, :, params['isLeaf'] > 0.5])
        elif i == 3:
            for j in range(l1):
                par = pca_params['Layer_{:d}_{:d}'.format(1, j)]
                l2 = par['isLeaf'].shape[0]- (int)(np.sum(par['isLeaf']))
                for k in range(l2):
                    if 'Layer_{:d}_{:d}_{:d}'.format(i - 1, j, k) not in pca_params:
                        continue
                    feature = feature_test['Layer_{:d}_{:d}_{:d}'.format(i - 1, j, k)]
                    par = pca_params['Layer_{:d}_{:d}_{:d}'.format(i-1, j, k)]
                    l3 = par['isLeaf'].shape[0]
                    for t in range(l3):
                        if 'Layer_{:d}_{:d}_{:d}_{:d}'.format(i, j, k, t) not in pca_params:
                            continue
                        params = pca_params['Layer_{:d}_{:d}_{:d}_{:d}'.format(i, j, k, t)]
                        fea_tmp = feature[:, :, :, t].reshape(feature.shape[0], feature.shape[1], feature.shape[2], 1)
                        no, output = tree(fea_tmp, params=params, train=0, bias=1, dilate=dilate[i])
                        feature_test['Layer_{:d}_{:d}_{:d}_{:d}'.format(i, j, k, t)] = output
                        feature_output.append(output[:, :, :, params['isLeaf'] > 0.5])
    return np.concatenate(feature_output, axis=3)

if __name__ == "__main__":
    import cv2
    X = cv2.imread('test.jpg')
    X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2])
    tmp, par = treePCA_train(X, energy_threshold=0.001, dilate = [1,2,3,4])
    print(tmp.shape)
    tmp1 = treePCA_test(X, par, dilate = [1,2,3,4])