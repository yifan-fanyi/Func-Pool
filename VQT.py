# v2020.05.17
# vector quantization tree

import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

class HierNode():
    def __init__(self, metric, isleaf=False, id='0'):
        self.kmeans = KMeans(n_clusters=2)
        self.metric = metric
        self.isleaf = isleaf
        self.center = []
        self.MSE = 0
        self.id = id

    def metric_(self, X):
        if 'func' in self.metric.keys():
            return self.metric['func'](X, self.metric)
        self.MSE = np.mean(euclidean_distances(np.mean(X, axis=0, keepdims=True), X))
        if self.MSE < self.metric['min_mse']:
            return True
        if X.shape[0] < 2 * self.metric['min_num_sample']:
            return True
        return False

    def fit(self, X, Y):
        if self.metric_(X) == True:
            self.isleaf = True
        if self.isleaf == False:
            self.kmeans.fit(X)
        self.center = np.mean(X, axis=0, keepdims=True)
        return self
    
    def predict(self, X):
        return self.kmeans.fit(X)
    
    def get_id(self):
        return int(self.id, base=2)

class VQT():
    def __init__(self, depth, metric):
        self.nodes = {}
        self.depth = depth
        self.num_node = 1
        self.metric = metric
        self.num_class = -1
        self.trained = False

    def fit(self, X, Y):
        self.num_class = len(np.unique(Y))
        tmp_data = [{'X':X, 'Y':Y, 'id':'0'}]
        for i in range(self.depth):
            tmp = []
            for j in range(len(tmp_data)):
                if tmp_data[j]['X'].shape[0] == 0:
                    continue
                tmp_node = HierNode(metric=self.metric, 
                                    isleaf=(i==self.depth-1), 
                                    id=tmp_data[j]['id'])
                tmp_node.fit(tmp_data[j]['X'], tmp_data[j]['Y'])
                label = tmp_node.predict(tmp_data[j]['X'])
                self.nodes[tmp_data[j]['id']] = copy.deepcopy(tmp_node)
                if tmp_node.isleaf == True:
                    continue
                for k in range(2):
                    idx = (label == k)
                    tmp.append({'X':tmp_data[j]['X'][idx], 'Y':tmp_data[j]['Y'][idx], 'id':tmp_data[j]['id']+str(k)})
                self.num_node += 1
            if len(tmp) == 0 and i != self.depth-1:
                print("       <Warning> depth %s not achieved, actual depth %s"%(str(self.depth), str(i+1)))
                self.depth = i
                break
            tmp_data = tmp
        print("       <INFP> Get %s Nodes!"%str(self.num_node))
        self.trained = True

    def check_continue_slow(self, X, MSE_target, parentID):
        mse = []
        for i in range(2):
            mse.append(euclidean_distances(X, self.nodes[parentID+str(i)].center))
        if np.min(mse) < MSE_target:
            return parentID + str(np.argmin(mse))
        elif self.nodes[parentID+str(i)].isleaf == True:
            return check_continue_slow(X, MSE_target, parentID+str(np.argmin(mse)))
        return parentID + str(np.argmin(mse))

    def quantize_slow(self, X, MSE_target):
        assert (self.trained == True), "Must call fit first!"
        tmp_pred = []
        parentID = '0'
        for i in range(X.shape[0]):
            tmp_pred.append(self.check_continue_slow(self, X, MSE_target, parentID))
        return tmp_pred

    def dequantize_slow(self, X):
        tmp_res = []
        for i in range(X.shape[0]):
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

if __name__ == "__main__":
    
    import cv2
    from skimage.util import view_as_windows

    X = cv2.imread('../Y.jpg', 0)
    print(X.shape)
    X = X.reshape(1, 512, 768, 1)
    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
