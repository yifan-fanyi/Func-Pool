# not sure why one symbol has one cdf 
# optimized for cnn not our case
# 2021.09.23
# x
import torchac
import numpy as np
import torch

class Arithmetic():
    def __init__(self):
        self.prob = None
        self.cdf = [0, ]
        
    def fit(self, idx):
        self.prob = np.zeros((len(np.unique(idx))))
        idx = idx.reshape(-1).astype('int16')
        for i in range(len(idx)):
            self.prob[idx[i]] += 1.
        self.prob /= len(idx)
        for i in range(len(self.prob)):
            self.cdf.append(self.cdf[-1]+self.prob[i])
        self.cdf[-1] = 1
        return self
    
    def gen_cdf(self, S):
        output_cdf = []
        for k in range(S[0]):
            for c in range(S[1]):
                for i in range(S[2]):
                    for j in range(S[3]):
                        output_cdf.append(self.cdf)
        output_cdf = np.array(output_cdf).reshape(S[0],S[1],S[2],S[3],len(self.cdf))
        return torch.from_numpy(output_cdf)
    
    def encode(self, idx):
        idx = np.moveaxis(idx, -1, 1).astype('int16')
        S = idx.shape
        byte_stream = torchac.encode_float_cdf(self.gen_cdf(S), 
                                               torch.from_numpy(idx), 
                                               check_input_bounds=True)
        return byte_stream, S

    def decode(self, byte_stream, S):
        sym_out = torchac.decode_float_cdf(self.gen_cdf(S), 
                                           byte_stream)
        idx = sym_out.numpy()
        idx = np.moveaxis(idx, 1, -1)
        return idx
    
    def check(self, sym, sym_out):
        assert np.sum(np.abs(sym-sym_out)) == 0, 'Error!'

if __name__ == "__main__":
    a = np.random.randint(0, 256, (2, 100, 100, 1))

    ac = Arithmetic().fit(a)

    b, s = ac.encode(a)
    d = ac.decode(b,s)
    ac.check(a, d)
    print('Avg', len(b)*8/(2*100*100))