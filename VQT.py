# v2020.05.22
# vector quantization tree

import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

class HierNode():
    def __init__(self, metric, isleaf=False, id='0', num_split=2):
        self.kmeans = KMeans(n_clusters=num_split)
        self.num_split = num_split
        self.metric = metric
        self.isleaf = isleaf
        self.center = []
        self.MSE = np.Inf
        self.id = id

    def metric_(self, X):
        if 'func' in self.metric.keys():
            return self.metric['func'](X, self.metric)
        self.MSE = np.mean(euclidean_distances(np.mean(X, axis=0, keepdims=True), X))
        print("       <INFO> MSE on nodes: ", self.MSE)
        if self.MSE < self.metric['min_mse']:
            return True
        if X.shape[0] < self.num_split * self.metric['min_num_sample']:
            return True
        return False

    def fit(self, X):
        if self.metric_(X) == True:
            self.isleaf = True
        if self.isleaf == False:
            self.kmeans.fit(X)
        self.center = np.mean(X, axis=0, keepdims=True)
        return self
    
    def predict(self, X):
        if self.isleaf == False:
            return self.kmeans.predict(X)
        return np.zeros((X.shape[0]))
    

class VQT():
    def __init__(self, depth, metric, num_split=2):
        self.nodes = {}
        self.depth = depth
        self.num_node = 1
        self.num_split = num_split
        self.metric = metric
        self.num_class = -1
        self.trained = False

    def fit(self, X):
        tmp_data = [{'X':X, 'id':'0'}]
        for i in range(self.depth):
            tmp = []
            for j in range(len(tmp_data)):
                if tmp_data[j]['X'].shape[0] == 0:
                    continue
                tmp_node = HierNode(metric=self.metric, 
                                    isleaf=(i==self.depth-1), 
                                    id=tmp_data[j]['id'],
                                    num_split=self.num_split)
                tmp_node.fit(tmp_data[j]['X'])
                label = tmp_node.predict(tmp_data[j]['X'])
                self.nodes[tmp_data[j]['id']] = copy.deepcopy(tmp_node)
                if tmp_node.isleaf == True:
                    continue
                for k in range(self.num_split):
                    idx = (label == k)
                    tmp.append({'X':tmp_data[j]['X'][idx], 'id':tmp_data[j]['id']+str(k)})
                self.num_node += self.num_split
            if len(tmp) == 0 and i != self.depth-1:
                print("       <Warning> depth %s not achieved, actual depth %s"%(str(self.depth), str(i+1)))
                self.depth = i
                break
            tmp_data = tmp
        print("       <INFP> Get %s Nodes!"%str(self.num_node))
        self.trained = True

    def check_continue_1by1(self, X, MSE_target, parentID):
        mse = []
        for i in range(self.num_split):
            if parentID+str(i) in self.nodes.keys():
                mse.append(euclidean_distances(X, self.nodes[parentID+str(i)].center))
        if self.nodes[parentID].isleaf == True:
            return parentID
        if np.min(mse) < MSE_target:
            return parentID + str(np.argmin(mse))
        elif self.nodes[parentID+str(i)].isleaf == False:
            return self.check_continue_1by1(X, MSE_target, parentID+str(np.argmin(mse)))
        return parentID + str(np.argmin(mse))

    def quantize_1by1(self, X, MSE_target):
        assert (self.trained == True), "Must call fit first!"
        tmp_pred = []
        parentID = '0'
        for i in range(len(X)):
            tmp_pred.append(self.check_continue_1by1(X[i].reshape(1, -1), MSE_target, parentID))
        return tmp_pred

    def dequantize_1by1(self, X):
        tmp_res = []
        for i in range(len(X)):
            tmp_res.append(self.nodes[X[i]].center)
        return tmp_res

    '''
    def quantize(self, X, MSE):
        assert (self.trained == True), "Must call fit first!"
        tmp_pred = []
        tmp_data = [{'X':X, 'idx':np.arange(0, X.shape[0]), 'id':'0'}]
        for i in range(self.depth):
            tmp = []
            for j in range(len(tmp_data)):
                if tmp_data[j]['X'].shape[0] == 0:
                    continue
                if self.nodes[tmp_data[j]['id']].isleaf == True:
                    prob = self.nodes[tmp_data[j]['id']].predict(tmp_data[j]['X'])
                    tmp_pred.append({'prob':prob.reshape(prob.shape[0], -1), 'idx':tmp_data[j]['idx']})
                    continue
                label = self.nodes[tmp_data[j]['id']].predict(tmp_data[j]['X'])
                for k in range(2):
                    idx = (label == k)
                    tmp.append({'X':tmp_data[j]['X'][idx],
                                'idx':tmp_data[j]['idx'][idx],
                                'id':tmp_data[j]['id']+str(k)})
            tmp_data = tmp
        prob, idx = [], []
        for i in range(len(tmp_pred)):
            prob.append(tmp_pred[i]['prob'])
            idx.append(tmp_pred[i]['idx'])
        prob = np.concatenate(prob, axis=0)
        idx = np.concatenate(idx, axis=0)
        idx = np.argsort(idx)
        return prob[idx]
    '''

class VQT_4D(VQT):
    def __init__(self, depth, metric, num_split=2, shrinkArg=None, inv_shrinkArg=None):
        super().__init__(depth=depth, metric=metric, num_split=num_split)
        self.shrinkArg = shrinkArg
        self.inv_shrinkArg = inv_shrinkArg
    
    def fit(self, X):
        X = self.shrinkArg['func'](X, self.shrinkArg)
        X = X.reshape(-1, self.shrinkArg['win']**2)
        super().fit(X)
        return self

    def quantize(self, X, MSE_target):
        X = self.shrinkArg['func'](X, self.shrinkArg)
        S = X.shape
        X = X.reshape(-1, self.shrinkArg['win']**2)
        return super().quantize_1by1(X, MSE_target), S

    def dequantize(self, X, shape):
        X = super().dequantize_1by1(X)
        X = np.array(X)
        X = X.reshape(shape)
        return self.inv_shrinkArg['func'](X, self.inv_shrinkArg)

if __name__ == "__main__":
    import cv2
    from utli import *
    from framework.evaluate import *
    X = cv2.imread('/Users/alex/Desktop/proj/compression/src/y.jpg', 0)
    print('input shape: ', X.shape)
    X = X.reshape(1, 512, 768, 1)

    vqt = VQT_4D(depth=5, metric={'min_mse':1, 'min_num_sample':20}, num_split=2, shrinkArg={'func':Shrink, 'win':4}, inv_shrinkArg={'func':invShrink, 'win':4})
    vqt.fit(X)
    Y, S = vqt.quantize(X, 1)
    XX = vqt.dequantize(Y, S)
    
    print('PSNR', PSNR(XX[0].astype('uint8'), X[0]))
