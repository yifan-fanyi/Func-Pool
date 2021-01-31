# 2021.01.27
# @yifan
# 
import numpy as np

from framework.core.myHVS_Q import compute_quantization_matrix

def Quant_Matrix(N, mode):
    JPEG_Q = np.array([16, 11, 10, 16, 24, 40, 51, 61,
                      12, 12, 14, 19, 26, 58, 60, 55,
                      14, 13, 16, 24, 40, 57, 69, 56,
                      14, 17, 22, 29, 51, 87, 80, 62,
                      18, 22, 37, 56, 68, 109, 103, 77,
                      24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101,
                      72, 92, 95, 98, 112, 100, 103, 99], dtype='float64')
    HVS_Q = np.array([16, 16, 16, 16, 17, 18, 21, 24,
                        16, 16, 16, 16, 17, 19, 22, 25,
                        16, 16, 17, 18, 20, 22, 25, 29,
                        16, 16, 18, 21, 24, 27, 31, 36,
                        17, 17, 20, 24, 30, 35, 41, 47,
                        18, 19, 22, 27, 35, 44, 54, 65,
                        21, 22, 25, 31, 41, 54, 70, 88,
                        24, 25, 29, 36, 47, 65, 88, 115], dtype='float64') # from Wang, Ching-Yang, Shiuh-Ming Lee, and Long-Wen Chang. "Designing JPEG quantization tables based on human visual system." Signal Processing: Image Communication 16.5 (2001): 501-506.
    if mode == 'JPEG':
        new_Q = JPEG_Q
    elif mode == 'HVS':
        new_Q = HVS_Q
    if N >= 50:   
        newQ = (100. - N) / 50. * new_Q.copy()
    else:
        newQ = 50. / N * new_Q.copy()
    return newQ

def Q(tPCA, X, Qstep, mode):
    if mode == 0:
        _, Qmatrix = compute_quantization_matrix(tPCA.K1, tPCA.K2, Qstep, win=8)
        Qmatrix = Qmatrix.reshape(-1)
        return np.round(X/Qmatrix)
    elif mode == 1:
        return np.round(X/Qstep)
    elif mode == 2:
        Qmatrix = Quant_Matrix(Qstep, 'JPEG')
        return np.round(X/Qmatrix)
    elif mode == 3:
        Qmatrix = Quant_Matrix(Qstep, 'HVS')
        return np.round(X/Qmatrix)
    elif mode == 'H':
        print('  mode=0: Ours')
        print('  mode=1: Constant Scalar')
        print('  mode=2: JPEG Quantization Matrix')
        print('  mode=3: HVS Quantization Matrix')
    else:
        return X
    
def dQ(tPCA, X, Qstep, mode):
    if mode == 0:
        _, Qmatrix = compute_quantization_matrix(tPCA.K1, tPCA.K2, Qstep, win=8)
        Qmatrix = Qmatrix.reshape(-1)
        return np.round(X*Qmatrix)
    elif mode == 1:
        return np.round(X*Qstep)
    elif mode == 2:
        Qmatrix = Quant_Matrix(Qstep, 'JPEG')
        return np.round(X*Qmatrix)
    elif mode == 3:
        Qmatrix = Quant_Matrix(Qstep, 'HVS')
        return np.round(X*Qmatrix)
    else:
        return X
    