import numpy as np
from numba import jit,prange

@jit(nopython=True)
def numba_unique(a,q):
    count = 0
    for x in a.ravel():
        if q[x] == 0:
            q[x] = 1
            count += 1
    return count
        
@jit(nopython=True)
def numba_subset_mean_along_axis(smean,pix,subset):
    # smean.shape = (D,S)
    # pix.shape = (T,D,S)
    # subset.shape = (T,S)
    T,D,S = pix.shape
    for d in prange(D):
        for s in prange(S):
            smean[d,s] = 0
            for t in prange(T):
                smean[d,s] += pix[subset[t,s],d,s]
            smean[d,s] /= T

@jit(nopython=True)
def numba_subset_mean_along_axis_mat(smean,pix,subset):
    # smean.shape = (D,B,S)
    # pix.shape = (T,D,S)
    # subset.shape = (T,D,B,S)
    T,D,B,S = subset.shape
    for d in prange(D):
        for s in prange(S):
            for b in prange(B):
                smean[d,b,s] = 0
                for t in range(T):
                    smean[d,b,s] += pix[subset[t,d,b,s],d,s]
                smean[d,b,s] /= T

@jit(nopython=True)
def numba_compute_muB(muB,pix,pix_smean,pix_mean,subset_B):
    # muB.shape = (B,S)
    # pix_mean.shape = (D,S)
    # pix_smean.shape = (D,S)
    # subset_B = (B,T,S)
    B,S = muB.shape
    for b in range(B):
        numba_subset_mean_along_axis(pix_smean,pix,subset_B[b])
        for s in range(S):
            delta = (pix_smean[:,s] - pix_mean[:,s])**2
            muB[b,s] = delta.mean()

@jit(nopython=True)
def numba_compute_deltas_bs(deltas,pix,pix_smean,pix_mean,subsets):
    # deltas.shape = (D,B,S)
    # pix.shape = (T,D,S)
    # pix_smean.shape = (D,B,S)
    # pix_mean.shape = (D,S)
    # subsets = (T,D,B,S)
    D,B,S = deltas.shape
    numba_subset_mean_along_axis_mat(pix_smean,pix,subsets)
    for d in prange(D):
        for b in prange(B):
            for s in prange(S):
                delta = (pix_smean[d,b,s] - pix_mean[d,s])**2
                deltas[d,b,s] = delta


