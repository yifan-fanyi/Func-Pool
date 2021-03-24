# 2021.03.21
# @yifan

import numpy as np

def int2bits(integer, lenth, is_uint, return_string=False):
    integer = (int)(integer)
    if is_uint == False:
        integer += pow(2, lenth-1)
    if integer >= pow(2, lenth):
        print("Warning dymanic range too large", integer, lenth)
    if integer >= 0:
        tmp = bin(integer)[2:].zfill(lenth)
    else:
        print('ERROR')
        exit()        
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

def bits2int(bits, is_uint):
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