# -- python imports --
import numpy as np
from einops import rearrange,repeat

# -- pytorch imports --
import torch

# -- project imports --
from align._utils import torch_to_numpy
from ._utils import per_pixel_centers,tile_to_ndims

def parse_inputs(isize,nframes,centers):
    print("Decrepcated.")
    if isize is None and centers is None:
        raise TypeError("Either isize or centers must not be None.")
    if not(centers is None): return centers
    else: return per_pixel_centers(isize,nframes)

def flow_to_pix(flow,centers=None,isize=None):

    # -- [old] get centers from inputs --
    # nframes = flow.shape[-2]+1
    # centers = parse_inputs(isize,nframes,centers)
    # centers = tile_to_ndims(centers,flow.shape[:-2])

    # -- [new] get centers from inputs --
    print("flow.shape",flow.shape)
    nframes = flow.shape[-2] + 1
    npix = flow.shape[-3] # new!
    if not(isize is None):
        assert npix == isize.h*isize.w,"Num of pixels should match."
    if centers is None:
        print("CREATING CENTER")
        centers = per_pixel_centers(isize,nframes)
    if centers.ndim == 2:
        centers = tile_to_ndims(centers,flow.shape[:-2])
    print("[post ndim check] ",centers.shape)

    # -- checking --
    assert centers.shape[-2] == flow.shape[-3], "num of pixels should match."
    # assert centers.shape[-1] == flow.shape[-3], "num of pixels should match."

    # -- [flow] reshape to 3-dim --
    shape = list(flow.shape)
    nframes_minus_1 = shape[-2]
    flow = flow.reshape(-1,nframes_minus_1,2)
    nsamples,nframes_minus_1,two = flow.shape

    # -- [centers] reshape to 2-dim --
    cshape = list(centers.shape)
    centers = centers.reshape(-1,2)
    nsamples,two = centers.shape

    # -- create blocks --
    print("flow.shape",flow.shape)
    pix = flow_to_pix_torch(flow,centers)
    print("centers.shape",centers.shape)
    print("pix.shape",pix.shape)

    # -- expand shape --
    pix_shape = shape
    pix_shape[-2] += 1
    pix = pix.reshape(pix_shape)

    # -- to tensor --
    pix = pix.type(torch.long)

    return pix

def flow_to_pix_torch(flow,centers):

    # -- compute deltas to ref --
    nsamples,nframes_minus_1,two = flow.shape
    nframes = nframes_minus_1 + 1
    ref_frame = nframes // 2

    # -- init pix --
    flip,csum = torch.fliplr,torch.cumsum
    zeros = torch.zeros((nsamples,1,2))
    left_idx = slice(None,nframes//2)
    right_idx = slice(nframes//2,None)
    left = -flip(csum(flip(flow[:,left_idx]),1))
    right = csum(flow[:,right_idx],1)
    pix = torch.cat([left,zeros,right],dim=1)
                         
    # -- add locations --
    print("[pre rep] centers.shape",centers.shape)
    centers = repeat(centers,'s two -> s t two',t=nframes)

    # -- create pix --
    pix += centers

    return pix
