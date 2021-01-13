# 2020.04.25
# dct based operation
import numpy as np
from skimage.util import view_as_windows

from transform import DCT

# zig zag scanning
class ZigZag():
    def __init__(self, N=8):
        self.N = N
        self.idx = self.zig_zag_getIdx(N).astype('int32')
        
    def zig_zag(self, i, j, n):
        if i + j >= n:
            return n * n - 1 - self.zig_zag(n - 1 - i, n - 1 - j, n)
        k = (i + j) * (i + j + 1) // 2
        return k + i if (i + j) & 1 else k + j

    def zig_zag_getIdx(self, N):
        idx = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                idx[i, j] = self.zig_zag(i, j, N)
        return idx.reshape(-1)
    
    def transform(self, X):
        S = list(X.shape)
        X = X.reshape(X.shape[0], -1)
        return X[:, np.argsort(self.idx)].reshape(S)
    
    def inverse_transform(self, X):
        S = list(X.shape)
        X = X.reshape(X.shape[0], -1)
        return X[:, self.idx].reshape(S)

# 8x8 quantization in JPEG
def JPEG_Quant(X, N=50, deQ=False):
    JPEG_Q = np.array([16, 11, 10, 16, 24, 40, 51, 61,
                      12, 12, 14, 19, 26, 58, 60, 55,
                      14, 13, 16, 24, 40, 57, 69, 56,
                      14, 17, 22, 29, 51, 87, 80, 62,
                      18, 22, 37, 56, 68, 109, 103, 77,
                      24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101,
                      72, 92, 95, 98, 112, 100, 103, 99], dtype='float64')
    if N > 50:   
        newQ = (100. - N) / 50. * JPEG_Q.copy()
    elif N < 50:
        newQ = 50. / N * JPEG_Q.copy()
    else:
        newQ = JPEG_Q.copy()
    if deQ == False:
        X /= newQ.reshape(64)
    else:
        X *= newQ.reshape(64)
    return np.round(X)

def JPEG(Xraw, N=50):
    X = Shrink(Xraw-128, {'win':8})
    tX = DCT(8,8).transform(X)
    tX = JPEG_Quant(tX, N=N, deQ=False)
    c, _ = zero_percent(tX)
    tX = JPEG_Quant(tX, N=N, deQ=True)
    iX = DCT(8,8).inverse_transform(tX)
    iX = invShrink(iX+128, {'win':8})
    return iX, c

# use DCT and IDCT for interpolation 
# input for Smooth_Interpolation: (?, initN, initN, 1)
# return for Smooth_Interpolation: (?, targetN, targetN, 1)
class Smooth_Interpolation():
    def __init__(self, initN=8, targetN=16, mode='block'):
        self.initN = initN
        self.targetN = targetN
        self.mode = mode
        self.dct1 = DCT(N=initN, P=initN)
        self.dct2 = DCT(N=targetN, P=targetN)
        
    def add_zeros_zigzag(self, X):
        zigzag1 = ZigZag(N=self.initN)
        zigzag2 = ZigZag(N=self.targetN)
        X = zigzag1.transform(X).reshape(X.shape[0], -1)
        X = np.concatenate((X, np.zeros((X.shape[0], self.targetN**2 - self.initN**2))), axis=1)
        X = zigzag2.inverse_transform(X).reshape(X.shape[0], self.targetN, self.targetN, 1)
        return X

    def add_zeros_block(self, X):
        X = np.concatenate((X, np.zeros((X.shape[0], self.targetN - self.initN, X.shape[2], 1))), axis=1)
        X = np.concatenate((X, np.zeros((X.shape[0], X.shape[1], self.targetN - self.initN, 1))), axis=2)
        return X

    def remove_zigzag_(self, X):
        zigzag1 = ZigZag(N=self.initN)
        zigzag2 = ZigZag(N=self.targetN)
        X = zigzag1.transform(X).reshape(X.shape[0], -1)
        X = X[:, :self.targetN**2]
        X = zigzag2.inverse_transform(X).reshape(X.shape[0], self.targetN, self.targetN, 1)
        return X

    def remove_block_(self, X):
        X = X[:, :self.targetN, :self.targetN]
        return X

    def rescale_(self, X):
        X *= self.targetN / self.initN
        return X

    def transform(self, X):
        assert (X.shape[1] ==  X.shape[2]), "<Error> Input shape not match!"
        X = self.dct1.transform(X)
        if self.initN < self.targetN:
            if self.mode == 'zigzag':
                X = self.add_zeros_zigzag(X)
            elif self.mode == 'block':
                X = self.add_zeros_block(X)
        elif self.initN > self.targetN:
            if self.mode == 'zigzag':
                X = self.remove_zigzag_(X)
            elif self.mode == 'block':
                X = self.remove_block_(X)
        X = self.rescale_(X)
        X = self.dct2.inverse_transform(X)
        return X.reshape(X.shape[0], self.targetN, self.targetN, 1)

if __name__ == "__main__":
    from sklearn import datasets
    print(" > This is a test example for Smooth_Interpolation.")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))[:10]
    si = Smooth_Interpolation(initN=8, targetN=16, mode='block')

    Y = si.transform(X)

    import matplotlib.pyplot as plt 
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(X[0,:,:,0])
    plt.title('Before interpolation')
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(Y[0,:,:,0])
    plt.title('After interpolation')

    plt.show()
