# -- python imports --
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F

def convert_edict(pix_data):
    if not isinstance(pix_data,edict):
        tmp = pix_data
        pix_data = edict()
        pix_data.pix = tmp
        pix_data.ftr = tmp
        pix_data.shape = tmp.shape
    return pix_data

def zero_out_of_bounds_pix(tile,patchsize,nblocks):
    r"""

    "tile_patches" uses "reflect" padding to ensure patchsize
    contains necessary non-zero pixels.

    We include "extra padding" (the "nblock" search space) for
    combinatorial optimization to grab associated offsets
    e.g. for patch[i,j] we want to grab patch[i+di,j+dj]
         for di,dj \in \{0,\ldot,nblocks\}
    
    We want this "extra padding" to be zero outside of the bounary.
    """
    raise NotImplemented("")
    

def tile_patches(burst,patchsize,pxform=None):
    """
    prepares a sequence of patches centered at each pixel location

    burst.shape = (T,B,C,H,W)

    pxform: 
    - a transformation to be applied to each patch
    - expects input of form: (npatches,c,h,w)
    """
    
    # -- backward compat --
    pix_only = (not isinstance(burst,edict))
    burst = convert_edict(burst)

    # -- init --
    T,B = burst.shape[:2]
    ps = patchsize
    unfold = nn.Unfold(ps,1,0,1)
    patches = edict()

    # -- tile pixel patches --
    pix_pad = rearrange(burst.pix,'t b c h w -> (b t) c h w')
    pix_pad = F.pad(pix_pad,(ps//2,ps//2,ps//2,ps//2),mode='reflect')
    patches.pix = unfold(pix_pad)
    shape_str = '(b t) (c ps1 ps2) r -> b t r (ps1 ps2 c)'
    patches.pix = rearrange(patches.pix,shape_str,t=T,ps1=ps,ps2=ps)

    # -- tile feature patches --
    if 'ftr' in burst:
        ftr_pad = rearrange(burst.ftr,'t b c h w -> (b t) c h w')
        ftr_pad = F.pad(ftr_pad,(ps//2,ps//2,ps//2,ps//2),mode='reflect')
        patches.ftr = unfold(ftr_pad)
        patches.ftr = rearrange(patches.ftr,shape_str,b=B,ps1=ps,ps2=ps)
    else:
        if pix_only and (pxform is None):
            patches.ftr = patches.pix
        else:
            _,_,R,_ = patches.pix.shape
            shape_str = 'b t r (ps1 ps2 c) -> (b t r) c ps1 ps2'
            pxform_inputs = rearrange(patches.pix,shape_str,ps1=ps,ps2=ps)
            features = pxform(pxform_inputs) # (b t r) nftrs 
            patches.ftr = rearrange(patches.ftr,'(b t r) f -> b t r f',b=B,t=T)

    # -- contiguous for faiss --
    patches.pix = patches.pix.contiguous()
    patches.ftr = patches.ftr.contiguous()
    patches.shape = patches.pix.shape

    # -- return for backward compat --
    return patches
