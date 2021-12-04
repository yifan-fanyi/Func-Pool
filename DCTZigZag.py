
import numpy as np
from dct import *
from ZigZag import *

class DCTZigZag():
    def __init__(self, n_components, zigzag_ratio=[0.33,0.33,0.33]):
        self.n_components = n_components
        self.C = len(zigzag_ratio)
        self.dct = None
        self.zigzag = ZigZag()
        self.zigzag_ratio = zigzag_ratio
        
    def fit(self, X):
        return self
    
    def transform(self, X):
        self.N = (int)(np.sqrt((X.shape[-1] / self.C)))
        self.dct = DCT3D(self.N, self.N, self.C)
        X = self.dct.transform(X)
        tmp = []
        c = (int)(self.N**2)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], self.C, c)
        for i in range(self.C):
            a = self.zigzag.transform(X[:,:,:,i,:])
            a = a[:,:,:,:min((int)(self.n_components*self.zigzag_ratio[i]), c)]
            tmp.append(a)
        return np.concatenate(tmp, axis=-1)
    
    def inverse_transform(self, X):
        tmp = []
        last = 0
        c = (int)(self.N**2)
        for i in range(self.C):
            n_comp = min((int)(self.n_components*self.zigzag_ratio[i]), c)
            t = X[:,:,:,last:last+n_comp]
            last += n_comp
            if c - n_comp > 0:
                t = np.concatenate((t, np.zeros((t.shape[0], t.shape[1], t.shape[2], c - n_comp))), axis=-1)
            t = self.zigzag.inverse_transform(t)
            tmp.append(t)
        X = np.concatenate(tmp, axis=-1)
        return self.dct.inverse_transform(X)

if __name__ == "__main__":
    from load_img import Load_from_Folder, Load_Images
    from util import *

    a = Load_from_Folder('/Users/alex/Desktop/proj/compression/data/test_256/', 'RGB', 5)
    a = np.array(a).astype('float32')
    a = Shrink(a, win=8)
    print(a.shape)
    dct = DCTZigZag(n_components=1000)
    b = dct.transform(a)
    print(b.shape, 'b')
    c = dct.inverse_transform(b)
    print(a.shape, c.shape)
    print(PSNR(a, c))
    c = invShrink(c, win=8)
    plt.imshow(c[0]/255)