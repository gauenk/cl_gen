import torch
import numpy as np
import numba
from numba import cuda
from numba.typed import List
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_uniform_float32
from numba import njit,jit,prange
from pyutils import torch_to_numpy

def compute_bootstrap(samples,scores_t,counts_t,ave,subsets,nbatches,batchsize):
    samples = torch_to_numpy(samples)    
    scores_t = torch_to_numpy(scores_t)
    counts_t = torch_to_numpy(counts_t)
    ave = torch_to_numpy(ave)
    subsets = torch_to_numpy(subsets)
    bootstrap_numba(samples,scores_t,counts_t,ave,subsets,nbatches,batchsize)

@cuda.jit
def create_weights_cuda(rng_states,nsubsets,nframes,weights):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    # cuda.gridDim
    tidx = cuda.grid(1)
    # tidx = tx + ty * bw
    # tidx,t,t2,subsize = 0,0,0,0
    to_add = False
    subset = cuda.local.array(shape=15,dtype=numba.uint8)
    # subset = cuda.local.array(shape=numba.int32(nframes),dtype=numba.float64)
        
    if tidx < weights.shape[0]:

        # -- init to zero --
        for t in range(nframes):
            subset[t] = 0

        # -- sample choice from uniform --
        for t in range(nframes):
            rand_float = nframes*xoroshiro128p_uniform_float32(rng_states, tidx)
            rand_t = 0
            while rand_t < rand_float:
                rand_t += 1
            rand_t -= 1
            subset[rand_t] = 1

        # -- count unique --
        nunique = 0
        for t in range(nframes):
            nunique += subset[t]

        # -- init weights --
        for t in range(nframes):
            weights[tidx,t] = -1./nframes

        # -- sample uniform --
        for t in range(nframes):
            rand_t = subset[t]
            if rand_t == 1:
                weights[tidx,t] = 1./nunique - 1./nframes
        

def create_weights(nsubsets,nframes):
    assert nframes <= 15, "Number of frames is maxed at 15."
    weights = np.zeros((nsubsets,nframes))
    threads_per_block = 64
    blocks = nsubsets//threads_per_block + 1
    rng_states = create_xoroshiro128p_states(blocks*threads_per_block,seed=123)
    create_weights_cuda[blocks,threads_per_block](rng_states,nsubsets,nframes,weights)
    weights = torch.FloatTensor(weights)
    return weights

@njit
def bootstrap_numba(samples,scores_t,counts_t,ave,subsets,nbatches,batchsize):
    nsubsets = len(subsets)
    for bidx in range(nbatches):
        for sidx in prange(batchsize):
            subset = subsets[bidx][sidx]
            counts_t[subset] += 1
            subset_pix = samples[subset]

            subset_ave = compute_mean_dim0(subset_pix)
            delta = (subset_ave - ave)**2
            loss = compute_mean_dim0(delta)
        
            scores_t[subset] += loss

@njit
def compute_mean_dim0(ndarray):
    shape = ndarray.shape
    ndarray = ndarray.reshape(shape[0],-1)
    ndarray_ave = np_mean(ndarray,0)
    ndarray_ave = ndarray_ave.reshape(shape[1:])
    return ndarray_ave

@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@njit
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)
