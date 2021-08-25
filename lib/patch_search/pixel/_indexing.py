
# TODO: write general "gather_neq_dims" for "pyutils"

# -- python imports --
import torch

# -- numba imports --
import numba
from numba import jit,prange,cuda

def index_along_frames(patches,dframes):

    # -- set cuda device --
    device = patches.device
    gpuid = device.index
    numba.cuda.select_device(gpuid)

    # -- get shapes --
    naligns,npatches,nframes,nftrs = patches.shape
    naligns,npatches = dframes.shape
    fpatches = torch.zeros((naligns,npatches,nftrs))
    fpatches = fpatches.to(device)
    # print("naligns,npatches,nframes,nftrs",naligns,npatches,nframes,nftrs)

    # -- create numba cuda --
    dframes_nba = cuda.as_cuda_array(dframes)
    patches_nba = cuda.as_cuda_array(patches)
    fpatches_nba = cuda.as_cuda_array(fpatches)

    # -- exec indexing cuda-kernel --
    threads_per_block = (32,32)
    blocks_aligns = naligns//threads_per_block[0] + (naligns%threads_per_block[0] != 0)
    blocks_patches = npatches//threads_per_block[1] + (npatches%threads_per_block[1] != 0)
    blocks = (blocks_aligns,blocks_patches)
    index_along_frames_cuda[blocks,threads_per_block](fpatches_nba,patches_nba,dframes_nba)
    return fpatches

@cuda.jit
def index_along_frames_cuda(fpatches,patches,dframes):

    a_idx,p_idx = cuda.grid(2)
    naligns,npatches,nframes,nftrs = patches.shape

    if a_idx < naligns and p_idx < npatches:
        frame = dframes[a_idx,p_idx]
        for f in range(nftrs):
            fpatches[a_idx,p_idx,f] = patches[a_idx,p_idx,frame,f]
