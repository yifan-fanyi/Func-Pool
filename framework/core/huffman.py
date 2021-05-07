# 2021.05.03
# @yifan
#
import numpy as np
import huffman

class Huffman():
    def __init__(self, hist, bins=32):
        self.hist = hist
        self.dict = {}
        self.inv_dict = {}
        self.make_dict()
        
    def make_dict(self):
        tmp = []
        for i in range(len(self.hist)):
            tmp.append((str(i), self.hist[i]))
        self.dict = huffman.codebook(tmp)
        self.inv_dict = {v: k for k, v in self.dict.items()}
        
    def encode(self, X):
        X = X.reshape(-1).astype('int32')
        stream = ''
        for i in range(len(X)):
            stream += str(X[i])
        return stream

    def decode(self, stream):
        dX, last = [], 0
        for i in range(len(stream)):
            if stream[last:i] in self.inv_dict:
                dX.append(self.inv_dict[stream[last:i]])
                last = i
        return dX