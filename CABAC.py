# CABAC
# major c++ source file modified from:
#   https://github.com/christianrohlfing/ISScabac
# this is a python wrap based on CMake generated executable file
# 2021.10.04
# @yifan
# log is output to ./cache/log_cabac.txt
import os
import datetime

class CABAC():
    @staticmethod                
    def write(st, file):
        if type(st) != str and isinstance(st, list) == False:
            assert False, "Only accept list or str as input!"
        with open(file, 'w') as f:
            for i in range(len(st)):
                if st[i] not in [1, 0, '1', '0']:
                    assert False, "Input not binary!"
                if type(st[i]) == int:
                    f.write(str((int)(st[i])))
                elif type(st[i]) == str:
                    f.write(st[i])
                else:
                    f.write(str((int)(st[i])))

    @staticmethod            
    def read(file, tolist=True):
        with open(file, 'r') as f:
            st = f.read()
        if tolist == True:
            tmp = []
            for i in range(len(st)):
                if st[i] == '0':
                    tmp.append(0)
                else:
                    tmp.append(1)
            return tmp
        return st

    @staticmethod
    def encode(n_context, infile, compfile="./cache/comp.txt"):
        time = datetime.datetime.now()
        os.system("echo "+str(time)+" >> ./cache/log_cabac.txt")
        os.system("echo cmd: "+"./CABAC/bin/cabac -e "+str(n_context)+" "+infile+" "+compfile+" >> ./cache/log_cabac.txt")
        os.system("./CABAC/bin/cabac -e "+str(n_context)+" "+infile+" "+compfile+" >> ./cache/log_cabac.txt")
        os.system("echo  "+" >> ./cache/log_cabac.txt")

    @staticmethod         
    def decode(n_context, nbits, outfile, compfile="./cache/comp.txt"):
        time = datetime.datetime.now()
        os.system("echo "+str(time)+" >> ./cache/log_cabac.txt")
        os.system("echo cmd: "+"./CABAC/bin/cabac -d "+str(n_context)+" "+str(nbits)+" "+outfile+" "+compfile+" >> ./cache/log_cabac.txt")
        os.system("./CABAC/bin/cabac -d "+str(n_context)+" "+str(nbits)+" "+outfile+" "+compfile+" >> ./cache/log_cabac.txt")
        os.system("echo  "+" >> ./cache/log_cabac.txt")

    @staticmethod
    def fsize(file="./cache/comp.txt"):
        return os.stat(file).st_size*8

if __name__ == "__main__":
    import time
    import numpy as np
    a = np.random.randint(2, size=(10000)).tolist() +np.ones(10000).astype('int16').tolist() 

    CABAC.write(a, 'cache/tmp.txt')
    CABAC.encode(3, 'cache/tmp.txt', 'cache/tmpd.cabac')
    os.system("sleep 1")
    size = CABAC.fsize('cache/tmpd.cabac')

    CABAC.decode(3, len(a), 'cache/tmpd.txt', 'cache/tmpd.cabac')
    b = CABAC.read('cache/tmpd.txt', tolist=True)
    assert 0==(np.sum(np.abs(np.array(a)-np.array(b))))