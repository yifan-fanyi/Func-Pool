# 2021.03.24
# @yifan
# most probable mode for bitstream encoding

# codeword should be sorted maybe in a similar mannar like HEVC's angular prediction
# or it should be sorted based on the context? smooth -> texture
# based on statics, we got the probability 
import numpy as np
import copy
from io import bits2int, int2bits
from mylearner import myLearner

# encode uses idx with shape (H, W)
# if index in MPM_list, send the index in MPM_list
# else, send 3 and the index in the codeword_list which contains all other codeword expect those in MPM_list
class MPM():
    def __init__(self, n_codeword, MPM_init_list):
        self.n_MPM = len(MPM_init_list)
        self.n_codeword = n_codeword
        self.length = (int)(np.log2(n_codeword-4))+1
        self.MPM_init_list = MPM_init_list
        #self.MPM_init_list.sort()
        self.MPM_list = self.MPM_init_list
        self.codeword_list = []  # code word not in current MPM_list
        self.stream = ''         # encoding bit stream <string> with 0/1 in it
        self.ct = 0              # decoding bit position indicater
                    
    def get_MPM(self, prev_idx):
        ct = 2
        self.MPM_list = copy.deepcopy(self.MPM_init_list)
        for i in range(len(prev_idx)):
            if prev_idx[i] not in self.MPM_list:
                self.MPM_list[ct] = prev_idx[i]
                ct -= 1 
        self.MPM_list.sort()
        self.codeword_list = []
        for i in range(self.n_codeword):
            if i not in self.MPM_list:
                self.codeword_list.append(i)
                
    def encode_one_idx(self, prev_idx, cur_idx):
        self.get_MPM(prev_idx)
        for i in range(len(self.MPM_list)):
            if self.MPM_list[i] == cur_idx:
                return i, i
        for i in range(len(self.codeword_list)):
            if self.codeword_list[i] == cur_idx:
                return i, 3
        assert (False), 'err'

    def get_prev(self, idx, i, j):
        if i < 0 or j < 0 or i >= idx.shape[0] or j >= idx.shape[1]:
            return self.MPM_init_list[0]
        else:
            return idx[i, j]
    
    def encode(self, idx):
        self.stream = ''
        idx = idx.astype('int16')
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                prev_idx = [self.get_prev(idx, i-1, j-1),
                            self.get_prev(idx, i-1, j),
                            self.get_prev(idx, i, j-1)]
                val, mode = self.encode_one_idx(prev_idx, idx[i,j])
                self.stream += int2bits(mode, 2, is_uint=True, return_string=True)
                if mode == 3:                    
                    self.stream += int2bits(val, self.length, is_uint=True, return_string=True)
        return self.stream
    
    def decode_one_idx(self, prev_idx, mode):
        self.get_MPM(prev_idx)
        if mode == 3:
            idx = bits2int(self.stream[self.ct:self.ct+self.length], is_uint=True)
            self.ct += self.length
            return self.codeword_list[idx]
        else:
            return self.MPM_list[mode]
            
    def decode(self, stream, H, W):
        self.stream = stream
        idx = np.zeros((H, W))
        self.ct = 0
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                prev_idx = [self.get_prev(idx, i-1, j-1),
                            self.get_prev(idx, i-1, j),
                            self.get_prev(idx, i, j-1)]
                mode = bits2int(self.stream[self.ct:self.ct+2], is_uint=True)
                self.ct += 2
                idx[i,j] = self.decode_one_idx(prev_idx, mode)
        return idx.astype('int16')
    
# fit uses idx with shape (K, H, W)
# encode uses idx with shape (H, W)
# apply ML to the idea,
# use previous index as feature and current index as label to train the learner
# if predict correctly, send 0, else send 1 and the corret index
# fitting time is too large
class ML_MPM(MPM):
    def __init__(self, learner, n_codeword, MPM_init_list):
        super().__init__(n_codeword, MPM_init_list)
        self.learner = myLearner(learner, n_codeword)
        self.length = (int)(np.log2(n_codeword-1))+1
        
    def fit(self, idx):
        prev, cur = [], []
        idx = idx.astype('int16')
        for k in range(idx.shape[0]):
            for i in range(idx.shape[1]):
                for j in range(idx.shape[2]):
                    prev.append( [self.get_prev(idx[k], i-1, j-1),
                                  self.get_prev(idx[k], i-1, j),
                                  self.get_prev(idx[k], i, j-1),
                                  self.get_prev(idx[k], i-1, j+1)])
                    cur.append(idx[k,i,j]) 
        prev = np.array(prev).reshape(-1, 4)
        cur = np.array(cur).reshape(-1,1)
        self.learner.fit(prev, cur)
        print('fit score:', self.learner.score(prev, cur))
        print('fit score top3',self.learner.topNscore(prev, cur, 3))
        return self
        
    def encode_one_idx(self, prev_idx, cur_idx):
        prev_idx = np.array(prev_idx).reshape(1, len(prev_idx))
        pred = self.learner.predict(prev_idx).reshape(-1)
        if pred[0] == cur_idx:
            return 0, 0
        else:
            return cur_idx, 1    
    
    def encode(self, idx):
        self.stream = ''
        idx = idx.astype('int16')
        x, y = [], []
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                prev_idx = [self.get_prev(idx, i-1, j-1),
                            self.get_prev(idx, i-1, j),
                            self.get_prev(idx, i, j-1),
                            self.get_prev(idx, i-1, j+1)]
                x.append(prev_idx)
                y.append(idx[i,j])
        x, y = np.array(x), np.array(y).reshape(-1)
        px = self.learner.predict(x).reshape(-1)
        print('test score',self.learner.score(x, y))
        print('test score top3',self.learner.topNscore(x, y, 3))
        for i in range(len(px)):
            if px[i] == y[i]:
                mode = 0
            else:
                mode = 1
            self.stream += int2bits(mode, 1, is_uint=True, return_string=True)
            if mode == 1:                    
                self.stream += int2bits(y[i], self.length, is_uint=True, return_string=True)
        return self.stream
    
    def decode_one_idx(self, prev_idx):
        mode = bits2int(self.stream[self.ct:self.ct+1], is_uint=True)
        self.ct += 1
        if mode == 1:
            idx = bits2int(self.stream[self.ct:self.ct+self.length], is_uint=True)
            self.ct += self.length
            return idx
        else:
            pred = self.learner.predict(np.array(prev_idx).reshape(1,-1))
            return pred
            
    def decode(self, stream, H, W, raw_idx=None):
        self.stream = stream
        idx = np.zeros((H, W))
        self.ct = 0
        prev_idx = []
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                prev_idx = [self.get_prev(idx, i-1, j-1),
                            self.get_prev(idx, i-1, j),
                            self.get_prev(idx, i, j-1),
                            self.get_prev(idx, i-1, j+1)]
                idx[i,j] = self.decode_one_idx(prev_idx)
                if raw_idx is not None:
                    assert(idx[i,j] == raw_idx[i,j]), 'Decoding Error!'
        return idx.astype('int16')

class ML_MPM3(ML_MPM):
    def __init__(self, learner, n_codeword, MPM_init_list):
        super().__init__(learner, n_codeword, MPM_init_list)
        self.length = (int)(np.log2(n_codeword-4))+1

    def encode(self, idx):
        self.stream = ''
        idx = idx.astype('int16')
        x, y = [], []
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                prev_idx = [self.get_prev(idx, i-1, j-1),
                            self.get_prev(idx, i-1, j),
                            self.get_prev(idx, i, j-1),
                            self.get_prev(idx, i-1, j+1)]
                x.append(prev_idx)
                y.append(idx[i,j])
        x, y = np.array(x), np.array(y).reshape(-1)
        px = self.learner.predict_proba(x)
        idx = np.argsort(px, axis=1)
        self.MPM_list = idx[:, -3:]
        print('test score',self.learner.score(x, y))
        print('test score top3',self.learner.topNscore(x, y, 3))
        for i in range(len(self.MPM_list)):
            tmp = copy.deepcopy(self.MPM_list[i])
            tmp.sort()
            mode, c = 3, 0
            for j in range(len(tmp)):
                if tmp[j] == y[i]:
                    mode = j
                if y[i] > tmp[j]:
                    c += 1
            self.stream += int2bits(mode, 2, is_uint=True, return_string=True)
            if mode == 3:                  
                self.stream += int2bits(y[i]-c, self.length, is_uint=True, return_string=True)
        return self.stream
    
    def decode_one_idx(self, prev_idx):
        mode = bits2int(self.stream[self.ct:self.ct+2], is_uint=True)
        self.ct += 2
        prob = self.learner.predict_proba(np.array(prev_idx).reshape(1,-1)).reshape(-1)
        pred = np.argsort(prob)[-3:]
        pred.sort()
        if mode == 3:
            codeword = []
            for i in range(self.n_codeword):
                if i not in pred:
                    codeword.append(i)
            idx = bits2int(self.stream[self.ct:self.ct+self.length], is_uint=True)
            self.ct += self.length
            return codeword[idx]
        else:
            return pred[mode]
        
    def decode(self, stream, H, W, raw_idx=None):
        self.stream = stream
        idx = np.zeros((H, W))
        self.ct = 0
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                prev_idx = [self.get_prev(idx, i-1, j-1),
                            self.get_prev(idx, i-1, j),
                            self.get_prev(idx, i, j-1),
                            self.get_prev(idx, i-1, j+1)]
                idx[i,j] = self.decode_one_idx(prev_idx)
                if raw_idx is not None:
                    assert(idx[i,j] == raw_idx[i,j]), 'Decoding Error!'
        return idx.astype('int16')

def Check(idx, didx):
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            if idx[i,j] != didx[i,j]:
                err_list = 'idx: '+str(idx[i,j])+', become: '+str(didx[i,j])+', pos: ('+str(i)+','+str(j)+')'
                print('Error!')
                print(err_list)
                assert (False), "Not Match!"

class ML_MPMa(MPM):
    def __init__(self, learner, n_codeword, MPM_init_list):
        super().__init__(n_codeword, MPM_init_list)
        self.learner = myLearner(learner, n_codeword)
        self.length = (int)(np.log2(n_codeword-5))+1
        
    def fit(self, idx):
        prev, cur = [], []
        idx = idx.astype('int16')
        for k in range(idx.shape[0]):
            for i in range(idx.shape[1]):
                for j in range(idx.shape[2]):
                    prev.append( [self.get_prev(idx[k], i-1, j-1),
                                  self.get_prev(idx[k], i-1, j),
                                  self.get_prev(idx[k], i, j-1),
                                  self.get_prev(idx[k], i-1, j+1)])
                    cur.append(idx[k,i,j]) 
        prev = np.array(prev).reshape(-1, 4)
        cur = np.array(cur).reshape(-1,1)
        self.learner.fit(prev, cur)
        print('fit score:', self.learner.score(prev, cur))
        print('fit score top4',self.learner.topNscore(prev, cur, 4))
        return self  
        
    def encode_fail(self, sprob, cur_idx):
        a = sprob[-4:-1]
        a.sort()
        b = sprob[:-4]
        b.sort()  
        mode = 3
        for i in range(len(a)):
            if a[i] == cur_idx:
                mode = i
        self.stream += int2bits(mode, 2, is_uint=True, return_string=True)
        if mode == 3:
            for i in range(len(b)):
                if cur_idx == b[i]:
                    self.stream += int2bits(i, self.length, is_uint=True, return_string=True)
                    return
            assert (False), 'Error, missing codeword!'
        else:
            pass

    def encode(self, idx):
        self.stream = ''
        idx = idx.astype('int16')
        x, y = [], []
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                prev_idx = [self.get_prev(idx, i-1, j-1),
                            self.get_prev(idx, i-1, j),
                            self.get_prev(idx, i, j-1),
                            self.get_prev(idx, i-1, j+1)]
                x.append(prev_idx)
                y.append(idx[i,j])
        x, y = np.array(x), np.array(y).reshape(-1)
        prob = self.learner.predict_proba(x)
        sprob = np.argsort(prob, axis=1)
        px = sprob[:, -1].reshape(-1)
        print('test score',self.learner.score(x, y))
        print('test score top4',self.learner.topNscore(x, y, 4))
        for i in range(len(px)):
            if px[i] == y[i]:
                mode = 0
            else:
                mode = 1
            self.stream += int2bits(mode, 1, is_uint=True, return_string=True)
            if mode == 1:                    
                self.encode_fail(sprob[i], y[i])
        return self.stream

    def decode_fail(self, idx):
        mode = bits2int(self.stream[self.ct:self.ct+2], is_uint=True)
        self.ct += 2   
        if mode < 3:
            a = idx[-4:-1]
            a.sort()
            return a[mode]
        else:
            b = idx[:-4]
            b.sort()
            val =  bits2int(self.stream[self.ct:self.ct+self.length], is_uint=True)
            self.ct += self.length
            return b[val]

    def decode_one_idx(self, prev_idx):
        mode = bits2int(self.stream[self.ct:self.ct+1], is_uint=True)
        self.ct += 1
        prob = self.learner.predict_proba(np.array(prev_idx).reshape(1,-1)).reshape(-1)
        if mode == 1:
            idx = self.decode_fail(np.argsort(prob))
            return idx
        else:
            return np.argmax(prob)
            
    def decode(self, stream, H, W, raw_idx=None):
        self.stream = stream
        idx = np.zeros((H, W)).astype('int16')
        self.ct = 0
        prev_idx = []
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                prev_idx = [self.get_prev(idx, i-1, j-1),
                            self.get_prev(idx, i-1, j),
                            self.get_prev(idx, i, j-1),
                            self.get_prev(idx, i-1, j+1)]
                idx[i,j] = self.decode_one_idx(prev_idx)
                if raw_idx is not None:
                    assert((int)(idx[i,j]) == raw_idx[i,j]), 'Decoding Error!'+str(i)+'_'+str(j)+'_'+str((int)(idx[i,j]))+'_'+str(raw_idx[i,j])
        return idx.astype('int16')

