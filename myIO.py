# 2021.10.14
# @yifan

import numpy as np

class myIO():
    @staticmethod
    def int2chr(val):
        val = (int)(val)
        assert val >=0 and val < 256, "Value Error!"
        return chr(val)
    
    @staticmethod
    def chr2int(char):
        return (ord)(char)

    @staticmethod
    def bstr2chr(bst):
        assert len(bst) <= 8, "Length Error!"
        val = bstr2int(bst)
        return int2chr(val)

    @staticmethod
    def chr2bstr(char):
        val = chr2int(char)
        bst = int2bstr(val)
        return bst

    @staticmethod
    def int2bstr(val):
        bst = "{0:b}".format(val)
        while len(bst) < 8:
            bst = '0' + bst
        return bst

    @staticmethod
    def bstr2int(bst):
        val = 0
        while len(bst) < 8:
            bst += '0' 
        for i in range(len(bst)):
            if bst[i] == '0':
                val = (val<<1)
            elif bst[i] == '1':
                val = (val<<1) + 1
            else:
                assert False, "Value Error!"
        return val

    @staticmethod
    def blist2bstr(blist):
        bstr = ''
        for i in blist:
            if i == '0' or i == 0:
                bstr += '0'
            elif i == '1' or i == 1:
                bstr += '1'
            else:
                assert False, "Value Error!"
        return bstr

    @staticmethod
    def bstr2blist(bstr):
        blist = []
        for i in bstr:
            if i == '0':
                blist.append(0)
            elif i == '1':
                blist.append(1)
            else:
                assert False, "Value Error!"
        return blist

def int2bits(integer, lenth, is_uint=True, return_string=False):
    integer = (int)(integer)
    if is_uint == False:
        integer += pow(2, lenth-1)
    if integer >= pow(2, lenth):
        print("Warning dymanic range too large", integer, lenth)
    if integer >= 0:
        tmp = bin(integer)[2:].zfill(lenth)
    else:
        print('ERROR')
        assert False, "Dymanic range not correct!"        
    bools = []
    if return_string == True:
        return tmp
    for i in range(len(tmp)):
        if tmp[i] == '0':
            bools.append(False)
        else:
            bools.append(True)
    assert (len(bools) == lenth), '<error> lenth not match'
    return bools

def bits2int(bits, is_uint=True):
    tmp = 0
    lenth = len(bits)
    for b in bits:
        if b == True or b == '1':
            tmp = tmp * 2 + 1
        else:
            tmp = tmp * 2 + 0
    if is_uint == False:
        tmp -= pow(2, lenth-1)
    return (int)(tmp)

def myWrite(b, f):
    if type(b) == list:
        b = myIO.blist2bstr(b)
    for i in range(0, len(b), 8):
        st = b[i:i+8]
        ch = myIO.bstr2chr(st)
        f.write(ch)
    if len(b[i:]) > 0:
        ch = myIO.bstr2chr(st)
        f.write(ch)

def myRead(f, size, typ='bstr'):
    bst = ''
    for i in range(size):
        v = f.read()
        if type == 'bstr':
            bst += myIO.chr2bstr(f)
    if typ == 'blist':
        return myIO.bstr2blist(bst)
    return bst