# 2020.04.22
# transformation and quantization in H.264
import numpy as np

class Int_Transform():
    def __init__(self):
        self.Cf = np.array([[ 1,  1,  1,  1],
                            [ 2,  1, -1, -2],
                            [ 1, -1, -1,  1],
                            [ 1, -2,  2, -1]])
        self.Ci = np.array([[  1,    1,    1,    1],
                            [  1,  1/2, -1/2,   -1],
                            [  1,   -1,   -1,    1],
                            [1/2,   -1,    1, -1/2]])

    def transform(self, X):
        return np.dot(np.dot(self.Cf, X), np.transpose(self.Cf))

    def inverse_transform(self, Y):
        return np.dot(np.dot(np.transpose(self.Ci), Y), self.Ci)

class Quantization():
    def __init__(self, QP):
        self.QP = QP
        self.Mf_list = np.array([[[13107,  8066, 13107,  8066],
                                  [ 8066,  5243,  8066,  5243],
                                  [13107,  8066, 13107,  8066],
                                  [ 8066,  5243,  8066,   5243]],
                                  
                                 [[11916,  7490, 11916,  7490],
                                  [ 7490,  4660,  7490,  4660],
                                  [11916,  7490, 11916,  7490],
                                  [ 7490,  4660,  7490,  4660]],
                                  
                                 [[10082,  6554, 10082,  6554],
                                  [ 6554,  4194,  6554,  4194],
                                  [10082,  6554, 10082,  6554],
                                  [ 6554,  4194,  6554,  4194]],
                                  
                                 [[ 9362,  5825,  9362,  5825],
                                  [ 5825,  3647,  5825,  3647],
                                  [ 9362,  5825,  9362,  5825],
                                  [ 5825,  3647,  5825,  3647]],
                                  
                                 [[ 8192,  5243,  8192,  5243],
                                  [ 5243,  3355,  5243,  3355],
                                  [ 8192,  5243,  8192,  5243],
                                  [ 5243,  3355,  5243,  3355]],
                                  
                                 [[ 7282,  4559,  7282,  4559],
                                  [ 4559,  2893,  4559,  2893],
                                  [ 7282,  4559,  7282,  4559],
                                  [ 4559,  2893,  4559,  2893]]])
        self.Mf = self.Mf_list[QP % 6]
        self.V_list = np.array([[[10, 13, 10, 13],
                                  [13, 16, 13, 16],
                                  [10, 13, 10, 13],
                                  [13, 16, 13, 16]],

                                 [[11, 14, 11, 14],
                                  [14, 18, 14, 18],
                                  [11, 14, 11, 14],
                                  [14, 18, 14, 18]],

                                 [[13, 16, 13, 16],
                                  [16, 20, 16, 20],
                                  [13, 16, 13, 16],
                                  [16, 20, 16, 20]],

                                 [[14, 18, 14, 18],
                                  [18, 23, 18, 23],
                                  [14, 18, 14, 18],
                                  [18, 23, 18, 23]],

                                 [[16, 20, 16, 20],
                                  [20, 25, 20, 25],
                                  [16, 20, 16, 20],
                                  [20, 25, 20, 25]],

                                 [[18, 23, 18, 23],
                                  [23, 29, 23, 29],
                                  [18, 23, 18, 23],
                                  [23, 29, 23, 29]]])
        self.V = self.V_list[QP % 6]

    def transform(self, X):
        return X * self.Mf / pow(2, 15 + self.QP // 6)

    def inverse_transform(self, Y):
        return Y * self.V * pow(2, self.QP//6)

class H264_Trans_Quant():
    def __init__(self, QP=1):
        self.core = Int_Transform()
        self.quant = Quantization(QP=QP)
    
    def transform(self, X):
        W = self.core.transform(X)
        Y = self.quant.transform(W)
        return np.round(Y)
    
    def inverse_transform(self, Y):
        W = self.quant.inverse_transform(Y)
        X = self.core.inverse_transform(W)
        return np.round(X / 64)

if __name__ == "__main__":
    X = np.array([[ 5, 11,  8, 10],
                  [ 9,  8,  4, 12],
                  [ 1, 10, 11,  4],
                  [19,  6, 15,  7]])
    print('-> RAW')
    print(X)
    co = H264_Trans_Quant(QP=10)
    a = co.transform(X)
    print('-> Trans + Quant')
    print(a)
    d = co.inverse_transform(a)
    print('-> Inverse')
    print(d)

  
