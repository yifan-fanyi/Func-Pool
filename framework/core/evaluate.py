# 2020.04.02
# @yifan

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

def MSE(ref, X):
    return mean_squared_error(ref, X)

def PSNR(X, XX):
    return 20*np.log10(255/np.sqrt(mean_squared_error(XX, X)))