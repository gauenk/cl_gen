# -- python imports --
import numpy as np
from einops import rearrange,repeat

# -- numba imports --
from numba import jit,prange

# -- pytorch imports --
import torch

# -- project imports --
from pyutils import torch_to_numpy

def pix_to_flow(pix):

    # -- check sizes --
    nimages,npix,nframes,two = pix.shape

    # -- create blocks --
    pix = rearrange(pix,'i p t two -> (i p) t two')
    flow = pix_to_flow_main(pix)
    flow = rearrange(flow,'(i p) tm1 two -> i p tm1 two',i=nimages)

    # -- to tensor --
    flow = torch.LongTensor(flow)

    return flow

def pix_to_flow_main(pix):

    # -- compute deltas to ref --
    nframes = pix.shape[-2]
    ref_frame = nframes // 2
    pix_ref = repeat(pix[:,ref_frame],'s two -> s t two',t=nframes)
    delta = pix - pix_ref
    flow = delta.clone()

    # -- get flows-to-_reference_ not frames-to-neighbor --
    flow[:,:ref_frame] = flow[:,1:ref_frame+1] - flow[:,:ref_frame]
    flow[:,ref_frame+1:] = flow[:,ref_frame+1:] - flow[:,ref_frame:-1]

    # -- convert _image_ coords to _object_ coords
    flow[...,1] = -flow[...,1]

    # -- remove middle event --    
    left = flow[:,:ref_frame]
    right = flow[:,ref_frame+1:]
    flow = torch.cat([left,right],dim=1)

    return flow
    
