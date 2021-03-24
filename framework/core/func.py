import numpy as np
import math

def random_Cauchy(mu, S):
    y = np.random.uniform(0, 1, S)
    return mu*np.tan(np.pi * (y-0.5))

def Cauchy(x, mu, bias=0):
        return 1 / np.pi * mu / (mu**2 + (x-bias)**2)

def Laplacian(x, b, bias=0):
    return 1 / (2 * b) * np.exp(-np.abs(x-bias)/b)

def Exp(x, round=0):
    if round > 0:
        val = 0
        for i in range(round):
            val += np.power(x, i) / math.factorial(i)
        return val
    else:
        return np.exp(x)