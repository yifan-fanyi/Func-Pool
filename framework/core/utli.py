# 2021.03.18
#
import numpy as np

def Hist(X, bins=-1):
    if bins < 0:
        bins = (int)(2* np.max([np.max(X), np.abs(np.min(X))])+7)
    X = np.round(X.reshape(-1)).astype('int32')
    hist, _ = np.histogram(X, bins=bins)
    return np.arange(-(bins//2), bins//2+1, 1), hist/len(X)

def mySort(abc):
    def sortbyAxis(abc, axis):
        sorted_abc = []
        for i in range(len(abc)):
            tmp = []
            idx = np.argmin(abc[:,axis])
            sorted_abc.append(abc[idx])
            abc = np.delete(abc, idx, 0)
        return np.array(sorted_abc)
    for i in range(abc.shape[1]):
        if i > 0:
            start = 0
            for j in range(1, len(abc)):
                if abs(abc[start, i-1] - abc[j, i-1]) > 1e-10:
                    abc[start:j] = sortbyAxis(abc[start:j], i)
                    start = j 
                if j == len(abc)-1:
                    abc[start:j+1] = sortbyAxis(abc[start:j+1], i)
        else:
            abc = sortbyAxis(abc, 0)
    return abc  