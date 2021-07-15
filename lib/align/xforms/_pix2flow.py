# -- python imports --
import numpy as np
from einops import rearrange

# -- numba imports --
from numba import jit,prange

# -- pytorch imports --
import torch

# -- project imports --
from align._utils import torch_to_numpy

def pix_to_flow(pix):

    # -- [pix] reshape to 3-dim --
    shape = list(pix.shape)
    nframes = shape[-2]
    pix = pix.reshape(-1,nframes,2)
    nsamples,nframes,two = pix.shape

    # -- create blocks --
    flow = pix_to_flow_main(pix)

    # -- expand shape --
    flow = shape_flow_ndims(flow,shape[:2])

    # -- to tensor --
    flow = torch.LongTensor(flow)

    return flow

def pix_to_flow_main(pix):

    # -- compute deltas to ref --
    nframes = pix.shape[-2]
    ref_frame = nframes // 2
    delta = pix - pix[:,ref_frame]
    flow = delta.clone()

    # -- get flows-to-_reference_ not frames-to-neighbor --
    flow[:,:ref_frame] = flow[:,1:ref_frame+1] - flow[:,:ref_frame]
    flow[:,ref_frame+1:] = flow[:,ref_frame+1:] - flow[:,ref_frame:-1]

    # -- remove middle event --    
    left = flow[:,:ref_frame]
    right = flow[:,ref_frame+1:]
    flow = torch.cat([left,right],dim=1)

    return flow
    
