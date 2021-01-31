# 2021.01.27
# @zhanxuan & yifan
# quantization matrix design
# 
import numpy as np
import math

def compute_quantization_matrix(K1, K2, q, win=4):
    # input:
    #      R and C: horizontal and vertical frequencies
    #      q: stepsize
    # output:
    #      H: human visual frequency weignting matrix
    #      Q: quantization matrix (q/H)
    # Note:
    #      this function is only for 4x4 matrix, R and C should be 1x4 list/array
    #      if the dimesnion of matrix change, some parameters need to be changed
    def compute_RC(X, Z):
        # input:
        #      X and Z: horizontal and vertical kernels from 2D^2-PCA
        # output:
        #      R and C: horizontal and vertical frequencies
        R, C = [], []
        for j in range(X.shape[1]):
            f_tmp = 0
            for i in range(X.shape[0]):
                if i == 0:
                    continue
                if X[i, j]*X[i-1, j] < 0:  # there is a sign change, frequency += slope
                    f_tmp += abs(X[i, j])
                    f_tmp += abs(X[i-1, j])
            R.append(f_tmp) 
        for j in range(Z.shape[1]):
            f_tmp = 0
            for i in range(Z.shape[0]):
                if i == 0:
                    continue
                if Z[i, j]*Z[i-1, j] < 0:  # there is a sign change, frequency += slope
                    f_tmp += abs(Z[i, j])
                    f_tmp += abs(Z[i-1, j])
            C.append(f_tmp)
        return R, C
    u, v = compute_RC(K1, K2)
    #     get the maximum frequency
    max_1, max_2 = np.max(u), np.max(v)
    if max_1 >= max_2:
        f_max = max_1
    else:
        f_max = max_2
    #     compute horizontal and vertical discrete frequencies
    f_u = list(np.zeros(8))
    f_v = list(np.zeros(8))
    for i in range(win):
        f_u[i] = (u[i])/(0.25*2*(win))
        f_v[i] = (v[i])/(0.25*2*(win))
    for i in range(win):  # use small value to represent 0 to avoid dividing by 0
        if f_u[i] == 0:
            f_u[i] = 0.00000001
        if f_v[i] == 0:
            f_v[i] = 0.00000001
    #     compute f_uv
    f_uv = np.zeros([win, win])
    for i in range(win):
        for j in range(win):
            f_uv[i,j] = (math.pi)*(math.sqrt(f_u[i]*f_u[i]+f_v[j]*f_v[j]))/(180*math.asin(1/math.sqrt(1+512*512)))
    #     compute s_uv
    s_uv = np.zeros([win, win])
    for i in range(win):
        for j in range(win):
            s_uv[i,j] = 0.15*math.cos(4*math.atan(f_u[i]/f_v[j]))+1.7/2
    #     compute f_hat_uv
    ff_uv = np.zeros([win, win])
    for i in range(win):
        for j in range(win):
            ff_uv[i, j] = f_uv[i, j]/s_uv[i, j]
    #     compute H
    H_f = np.zeros([win, win])
    for i in range(win):
        for j in range(win):
            H_f[i, j] = 2.2*(0.192+0.114*ff_uv[i, j])*(math.exp(-1*math.pow((0.114*ff_uv[i, j]), 1.1)))
    for i in range(win):
        for j in range(win):
            if ff_uv[i, j] <= f_max:
                H_f[i, j] = 1 
    #     compute Q
    Q = np.round(q/H_f)
    return H_f, Q

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
                        24, 25, 29, 36, 47, 65, 88, 115], dtype='float64')
    if mode == 'JPEG':
        new_Q = JPEG_Q
    elif mode == 'HVS':
        new_Q = HVS_Q
    if N >= 50:   
        newQ = (100. - N) / 50. * new_Q.copy()
    else:
        newQ = 50. / N * new_Q.copy()
    return newQ
