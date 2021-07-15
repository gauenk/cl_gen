# -- python imports --
import numpy as np
from einops import rearrange,repeat

# -- pytorch imports --
import torch

# -- project imports --
from align._utils import torch_to_numpy
from ._utils import per_pixel_centers,tile_to_ndims

def parse_inputs(nimages,isize,centers):
    if isize is None and centers is None:
        raise TypeError("Either isize or centers must not be None.")
    if not(centers is None): return centers
    else:
        centers = per_pixel_centers(isize)
        centers = np.broadcast_to(centers,(nimages,) + centers.shape)
        return centers

def flow_to_pix(flow,centers=None,isize=None):

    # -- check shapes --
    nimages,npix,nframes_minus_1,two = flow.shape
    centers = parse_inputs(nimages,isize,centers)
    c_nimages,c_npix,two = centers.shape
    assert nimages == c_nimages,"num of images must be eq."
    assert npix == c_npix,"num of pixels must be eq."

    # -- create blocks --
    flow = rearrange(flow,'i p tm1 two -> (i p) tm1 two')
    centers = rearrange(centers,'i p two -> (i p) two')
    pix = flow_to_pix_torch(flow,centers)
    pix = rearrange(pix,'(i p) t two -> i p t two',i=nimages)

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

    # -- change from _spatial_ _object_ motion into _image coords_ _object_ motion
    flow[...,1] = -flow[...,1] 
    
    # -- swap dx and dy --
    

    r"""
    go from

        "x -> x -> x*" to get "sum(->,->), sum(->)"

    1. (1st flip) the first element is _further_ from ref than the left_idx[-1] element
    2. The cumulative sum goes from single arrow to sum of arrows
    3. (2nd flip) back to original order
    4. (negative) the origin of the starts from _ref_ location.
    """

    left = -flip(csum(flip(flow[:,left_idx]),1))
    right = csum(flow[:,right_idx],1)
    pix = torch.cat([left,zeros,right],dim=1)
                         
    # -- add locations --
    centers = repeat(centers,'s two -> s t two',t=nframes)

    # -- create pix --
    pix += centers

    return pix