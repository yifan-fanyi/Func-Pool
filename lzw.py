# 2021.09.26
# Lempel–Ziv–Welch Compression
# @yifan
#
import numpy as np 
import copy

class LZW():
    def __init__(self, init_dict):
        self.init_dict = init_dict
        self.inv_init_dict = {v: k for k, v in self.init_dict.items()}
        self.init_dict_size = len(self.init_dict.keys())

    def encode(self, data):
        workdict = copy.deepcopy(self.init_dict)
        nc = self.init_dict_size + 1          
        tmp = ""             
        cdata = []    
        for sym in data:                     
            tmp_sym = tmp + sym 
            if tmp_sym in workdict: 
                tmp = tmp_sym
            else:
                cdata.append(workdict[tmp])
                if(len(workdict) <= 256):
                    workdict[tmp_sym] = nc
                    nc += 1
                tmp = sym
        if tmp in workdict:
            cdata.append(workdict[tmp])
        return cdata

    def decode(self, cdata):
        workdict = copy.deepcopy(self.inv_init_dict)
        nc = self.init_dict_size + 1
        decdata = ""
        tmp = ""
        for code in cdata:
            if not (code in workdict):
                workdict[code] = tmp + (tmp[0])
            decdata += workdict[code]
            if not(len(tmp) == 0):
                workdict[nc] = tmp + (workdict[code][0])
                nc += 1
            tmp = workdict[code]
        return decdata

if __name__ == "__main__":
    lzw = LZW({'a':1, 'b':2,'c':3,'d':4})
    a = 'aaabcccdaa'
    print('Input   :', a)
    c = lzw.encode(a)
    print('Encoded :', c)
    d = lzw.decode(c)
    print('Decoded :', d)
    assert a == d, "Error!"