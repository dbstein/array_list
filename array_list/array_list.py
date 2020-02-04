import numpy as np
import numba
from numba import int64, typeof

def create_selector(L):
    CL = np.concatenate(L)
    NS = np.array([LL.shape[0] for LL in L])
    NS = np.concatenate([(0,), np.cumsum(NS)])
    
    cdtype = numba.typeof(CL)
    ndtype = numba.typeof(NS)
    spec = [
        ('CL', cdtype),
        ('NS', ndtype),
    ]

    @numba.jitclass(spec)
    class ArrayList(object):
        def __init__(self, CL, NS):
            self.CL = CL
            self.NS = NS
        def get(self, i):
            return self.CL[self.NS[i]:self.NS[i+1]]
            
    return ArrayList(CL, NS)
