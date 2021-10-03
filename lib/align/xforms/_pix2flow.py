# -- python imports --
import numpy as np
from einops import rearrange,repeat

# -- numba imports --
from numba import jit,prange

# -- pytorch imports --
import torch

# -- project imports --
from pyutils import torch_to_numpy

def pix_to_flow(pix,ftype="ref"):

    # -- check sizes --
    nimages,npix,nframes,two = pix.shape

    # -- create blocks --
    pix = rearrange(pix,'i p t two -> (i p) t two')
    if ftype == "ref":
        flow = pix_to_flow_ref(pix)
    elif ftype == "seq":
        flow = pix_to_flow_seq(pix)
    else:
        raise ValueError(f"Unknown flow type [{ftype}]")
    flow = rearrange(flow,'(i p) tm1 two -> i p tm1 two',i=nimages)

    # -- to tensor --
    flow = torch.LongTensor(flow)

    return flow

def pix_to_flow_ref(pix):

    # -- compute deltas to ref --
    nframes = pix.shape[-2]
    ref_frame = nframes // 2
    pix_ref = repeat(pix[:,ref_frame],'s two -> s t two',t=nframes)
    delta = pix_ref - pix
    delta[...,0] = -delta[...,0]

    # print(pix[:,1])
    # print(pix[230:240,0])
    # print(pix[230:240,1])
    # print(pix[:,1] - pix[:,0])
    # print(pix[230:240,1] - pix[230:240,0])

    # delta_tmp = (pix[:,1] - pix[:,2]).numpy()


    # print(torch.max(pix[:,0,:]))
    # print(torch.max(pix[:,2,:]))
    # print(torch.min(pix[:,0,:]))
    # print(torch.min(pix[:,2,:]))

    # print(torch.topk(pix[:,0,1],10))
    # print(torch.topk(pix[:,0,1],10,largest=False))
    # print(torch.topk(pix[:,2,1],10))
    # print(torch.topk(pix[:,2,1],10,largest=False))

    # print(torch.topk(pix[:,0,0],10))
    # print(torch.topk(pix[:,0,0],10,largest=False))
    # print(torch.topk(pix[:,2,0],10))
    # print(torch.topk(pix[:,2,0],10,largest=False))

    # print("\n\n\n\n\n\n DELTA\n\n\n\n\n\n\n")


    # print(torch.max(delta[:,0,:]))
    # print(torch.max(delta[:,2,:]))
    # print(torch.min(delta[:,0,:]))
    # print(torch.min(delta[:,2,:]))

    # print(torch.topk(delta[:,0,1],10))
    # print(torch.topk(delta[:,0,1],10,largest=False))
    # print(torch.topk(delta[:,2,1],10))
    # print(torch.topk(delta[:,2,1],10,largest=False))

    # print(torch.topk(delta[:,0,0],10))
    # print(torch.topk(delta[:,0,0],10,largest=False))
    # print(torch.topk(delta[:,2,0],10))
    # print(torch.topk(delta[:,2,0],10,largest=False))

    flow = delta.clone()

    return flow

def pix_to_flow_seq(pix):

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
    
