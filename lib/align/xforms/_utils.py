
# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF


#
# Proper Shaping
#

def tile_to_ndims(tensor,xtra_dims):
    shape = list(tensor.shape)
    if apply_xtra_dims(shape,xtra_dims):
        bshape = list(xtra_dims) + shape
        broadcast = np.broadcast_to(tensor,bshape)
        return broadcast

def per_pixel_centers(isize,nframes):
    npix = isize.h * isize.w
    shape = [npix,2]
    xy = np.c_[np.unravel_index(np.arange(npix),(isize.h,isize.w))]
    # xy = rearrange(xy,'p two -> p 1 two')
    # xy = np.broadcast_to(xy,shape)
    return xy

def create_isize(h,w):
    isize = edict({'h':h,'w':w})
    return isize

def shape_flow_ndims(flow,shape):
    postfix = list(flow.shape[-2:])
    print(shape,postfix,flow.shape)
    flow_shape = shape + postfix
    flow = flow.reshape(flow_shape)
    return flow[-2:]

def shape_blocks_3dims(blocks):
    shape = list(blocks.shape)
    nframes = shape[-1]
    blocks = blocks.reshape(-1,nframes)
    return blocks,shape[:-1]

def shape_burst_3dims(burst):
    shape = list(burst.shape)
    nframes = shape[-1]
    burst = burst.reshape(-1,nframes)
    return burst,shape[:-1]

def shape_pix_3dims(pix):
    shape = list(pix.shape)
    nframes = shape[-1]
    pix = pix.reshape(-1,nframes)
    return pix,shape[:-1]


def apply_xtra_dims(shape,xtra_dims):
    l_shape,l_xtra = len(shape),len(xtra_dims)
    if len(shape) >= len(xtra_dims):
        is_eq = np.isclose(xtra_dims,shape[l_xtra])
        return not(is_eq)
    else:
        return True

#
# Supporting
# 

def ensure_3d(blocks):
    if blocks.ndim == 2:
        return rearrange(blocks,'b t -> b 1 t')
    elif blocks.ndim == 3:
        return blocks
    else:
        raise ValueError(f"[align.xforms._utils.py]: Unknown block dims [{blocks.ndim}]")

def th_rcumsum(tensor,dim=0):
    return torch.flip(torch.cumsum(torch.flip(tensor,(dim,)),dim),(dim,))

def np_rcumsum(tensor,dim=0):
    return np.flip(np.cumsum(np.flip(tensor,axis=dim),axis=dim),axis=dim)

def reshape_and_pad(images,nblocks):
    T,B,C,H,W = images.shape
    images = rearrange(images,'t b c h w -> (t b) c h w')
    padded = F.pad(images, [nblocks//2,]*4, mode="reflect")
    padded = rearrange(padded,'(t b) c h w -> t b c h w',b=B)
    return padded

def tile_across_blocks(batches,nblocks):
    B = batches.shape[1]
    H = nblocks
    FS = batches.shape[-1]
    crops,tiles = [],[]
    grid = np.arange(2*nblocks**2).reshape(nblocks,nblocks,2)
    blocks = []
    center = H//2
    padded = reshape_and_pad(batches,nblocks)
    for dy in range(-H//2+1,H//2+1):
        for dx in range(-H//2+1,H//2+1):
            # grid[dy+H//2,dx+H//2,:] = (dy+center,dx+center)
            # print(grid[i+center,j+center],(i+center,j+center))
            crop = tvF.crop(padded,dy+center,dx+center,FS,FS)
            # endy,endx = dy+center+FS,dx+center+FS
            # crop = padded[...,dy+center:endy,dx+center:endx]
            crops.append(crop)
    # print(grid)
    # print(blocks)
    crops = torch.stack(crops,dim=2) # t b g c h w
    return crops
