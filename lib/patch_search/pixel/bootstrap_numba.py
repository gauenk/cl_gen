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
def create_weights_cuda(rng_states,nsubsets,nframes,weights,counts):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    # cuda.gridDim
    tidx = cuda.grid(1)
    # tidx = tx + ty * bw
    # tidx,t,t2,subsize = 0,0,0,0
    to_add = False
    subset = cuda.local.array(shape=20,dtype=numba.uint8)
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
            counts[tidx,t] = 0

        # -- sample uniform --
        for t in range(nframes):
            rand_t = subset[t]
            if rand_t == 1:
                weights[tidx,t] = 1./nunique - 1./nframes
                counts[tidx,t] += 1
        
@cuda.jit
def fill_weights_pix_cuda(rng_states,nsubsets,npix,nframes,weights,counts):

    # -- get kernel indices --
    r_idx = cuda.grid(1)
    s_idx,p_idx = cuda.grid(2)
    subset = cuda.local.array(shape=20,dtype=numba.uint8)
        
    if s_idx < weights.shape[0] and p_idx < weights.shape[1]:

        # -- init to zero --
        for t in range(nframes):
            subset[t] = 0

        # -- sample choice from uniform --
        for t in range(nframes):
            rand_float = nframes*xoroshiro128p_uniform_float32(rng_states, r_idx)
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
            weights[s_idx,p_idx,t] = -1./nframes

        # -- sample uniform --
        for t in range(nframes):
            rand_t = subset[t]
            if rand_t == 1:
                weights[s_idx,p_idx,t] = 1./nunique - 1./nframes
                counts[s_idx,p_idx,t] += 1


def fill_weights_pix(weights,counts,nsubsets,npix,nframes,gpuid):
    weights = weights
    counts = counts
    fill_weights_pix_cuda_launcher(weights,counts,nsubsets,npix,nframes,gpuid)
    return weights,counts

def fill_weights_pix_cuda_launcher(weights,counts,nsubsets,npix,nframes,gpuid):
    assert nframes <= 15, "Number of frames is maxed at 15."
    numba.cuda.select_device(gpuid)
    device = weights.device
    weights = numba.cuda.as_cuda_array(weights)
    counts = numba.cuda.as_cuda_array(counts)
    threads_per_block = (32,32)

    blocks_subsets = nsubsets//threads_per_block[0] + (nsubsets%threads_per_block[0] != 0)
    blocks_pix = npix//threads_per_block[1] + (npix%threads_per_block[1] != 0)
    blocks = (blocks_subsets,blocks_pix)

    nthreads = int(np.product([blocks[i] * threads_per_block[i] for i in range(2)]))
    seed = int(torch.rand(1)*100)
    rng_states = create_xoroshiro128p_states(nthreads,seed=seed)

    fill_weights_pix_cuda[blocks,threads_per_block](rng_states,nsubsets,npix,
                                                    nframes,weights,counts)
    return weights



def fill_weights(weights,counts,nsubsets,nframes,gpuid):
    weights = weights
    counts = counts
    fill_weights_foo(weights,counts,nsubsets,nframes,gpuid)
    return weights,counts
    # weights = weights.cpu()
    # weights = weights.numpy()
    # fill_weights_bar(weights,counts,nsubsets,nframes,gpuid)
    # weights = torch.FloatTensor(weights).to(gpuid)
    # return weights

@njit
def fill_weights_bar(weights,counts,nsubsets,nframes,gpuid):
    rands = np.random.choice(nframes,(nsubsets,nframes))
    for s in prange(nsubsets):
        uniques = np.unique(rands[s])
        n_uniques = len(uniques)
        for t in range(nframes):
            weights[s,t] = -1./nframes
        for t in uniques:
            weights[s,t] += 1./n_uniques
            counts[s,t] += 1
    return weights

def fill_weights_foo(weights,counts,nsubsets,nframes,gpuid):
    assert nframes <= 15, "Number of frames is maxed at 15."
    numba.cuda.select_device(gpuid)
    device = weights.device
    weights = numba.cuda.as_cuda_array(weights)
    counts = numba.cuda.as_cuda_array(counts)
    threads_per_block = 1024
    blocks = nsubsets//threads_per_block + (nsubsets%threads_per_block != 0)
    seed = int(torch.rand(1)*100)
    rng_states = create_xoroshiro128p_states(blocks*threads_per_block,seed=seed)
    create_weights_cuda[blocks,threads_per_block](rng_states,nsubsets,nframes,
                                                  weights,counts)
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
