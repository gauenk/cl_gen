# -- python imports --
import numpy as np
from einops import rearrange

# -- numba imports --
from numba import jit,prange

# -- pytorch imports --
import torch
import torch.nn.functional as F

# -- project imports --
# from align.xforms._global_motion import *
from align.xforms._utils import *
from pyutils import torch_to_numpy
# from ._flow2blocks import flow_to_blocks
from ._flow2pix import flow_to_pix
# from ._pix2blocks import pix_to_blocks
from ._blocks2pix import blocks_to_pix


def align_from_flow(burst,flow,nblocks,centers=None,isize=None):
    pix = flow_to_pix(flow,centers,isize)
    return align_from_pix(burst,pix,nblocks)

def align_from_blocks(burst,blocks,nblocks,centers=None,isize=None):
    pix = blocks_to_pix(blocks,nblocks,centers,isize)
    return align_from_pix(burst,pix,nblocks)

def align_from_pix(burst,pix,nblocks):
    r"""
    Align a burst of frames 
    from a pixel tensor representing 
    each pixel's adjustment
    """

    # -- shapes --
    nframes,nimages,c,h,w = burst.shape
    nimages,npix,nframes,two = pix.shape

    # -- add paddings --
    burst = rearrange(burst,'t i c h w -> (t i) c h w')
    pad_burst = F.pad(burst,[nblocks//2,]*4,mode='reflect')
    pad_burst = rearrange(pad_burst,'(t i) c h w -> t i c h w',i=nimages)

    # -- ensure ndarray --
    pad_burst = torch_to_numpy(pad_burst)
    pix = torch_to_numpy(pix)

    # -- create blocks --
    aligned = np.zeros((nframes,nimages,c,h,w)).astype(np.float64)
    pad_burst = pad_burst.astype(np.float64)
    align_from_pix_numba(aligned,pad_burst,pix,nblocks)

    # -- back to torch --
    aligned = torch.FloatTensor(aligned)

    return aligned

def verify_burst_pix_sizes(burst,pix):
    npix = np.product(burst.shape[-2:])
    nsamples = pix.shape[0]
    assert nsamples % npix == 0,"Must be a multiple of number of pixels."

@jit(nopython=True)
def align_from_pix_numba(aligned,burst,pix,nblocks):
    nframes,nimages,c,h_pad,w_pad = burst.shape
    h = h_pad - 2*(nblocks//2)
    w = w_pad - 2*(nblocks//2)
    npix = h*w
    ref_t = nframes//2
    for i in range(nimages):
        for t in range(nframes):
            for p in range(npix):
                ref_xy =  pix[i,p,ref_t] #ordering of "p" is not always order of pix.
                # r_col,r_row = ref_xy[0],ref_xy[1]
                r_row,r_col = p//w,p%w
                # print(r_col,r_row,re_col,re_row)
                xy_xfer = pix[i,p,t] + nblocks//2 # padding
                b_col,b_row = xy_xfer[0],xy_xfer[1]
                aligned[t,i,:,r_row,r_col] = burst[t,i,:,b_row,b_col]


# @jit(nopython=True)
# def align_from_blocks_numba(crops,burst,pix,nblocks):
#     nsamples,nframes = patches.shape[:2]
#     c,h,w = patches.shape[2:]
#     ps,ref_t = int(nblocks),nframes//2

#     for s in prange(nsamples):
#         for t in prange(nframes):
#             if t == ref_t:
#                 hs = int(nblocks//2)
#                 ws = int(nblocks//2)
#             else:
#                 t_corr = t if t < ref_t else t - 1
#                 pix_t = pix[s,t_corr]
#                 hs = int(block // nblocks)
#                 ws = int(block % nblocks)
#             he,we = hs + ps,ws + ps
#             # crops[s,t,:,:,:] = patches[s,t,:,hs:he,ws:we]
#             crops[s,t,:,:,:] = patches[s,t,:,hs:he,ws:we]


# def align_burst_from_block(burst,block,nblocks,mtype):
#     if mtype == "global":
#         return align_burst_from_block_global(burst,block,nblocks)
#     elif mtype == "local":
#         return align_burst_from_block_local(burst,block,nblocks)
#     else:
#         raise ValueError(f"Uknown motion type [{mtype}]")

# def align_burst_from_flow(burst,flow,nblocks,mtype):
#     if mtype == "global":
#         return align_burst_from_flow_global(burst,flow,nblocks)
#     elif mtype == "local":
#         raise NotImplemented("No local flow supported yet.")
#     else:
#         raise ValueError(f"Uknown motion type [{mtype}]")

# def align_burst_from_block_global(bursts,blocks,nblocks):
#     T,B,C,H,W = bursts.shape
#     # T,B,FS = bursts.shape[0],bursts.shape[1],bursts.shape[-1]
#     ref_t = T//2
#     tiles = tile_across_blocks(bursts,nblocks)
#     crops = []
#     for b in range(B):
#         for t in range(T):
#             index = blocks[b,t].item()
#             crops.append(tiles[t,b,index])
#     crops = rearrange(crops,'(b t) c h w -> t b c h w',b=B)
#     return crops

# def align_burst_from_flow_padded_patches(patches,flow,nblocks,patchsize):
#     nimages,nsegs = flow.shape[:2]
#     flow = torch.LongTensor(rearrange(flow,'i s t two -> (i s) t two'))
#     print("[flow.shape]: ",flow.shape)
#     ref_blocks = global_flow_to_blocks(flow,nblocks)
#     blocks = global_blocks_ref_to_frames(ref_blocks,nblocks)
#     blocks = rearrange(blocks,'(i s) t two -> i s t two',i=nimages)
#     print("blocks.shape",blocks.shape)
#     patches = torch_to_numpy(patches)
#     blocks = torch_to_numpy(blocks)
#     return align_burst_from_blocks_padded_patches(patches,blocks,nblocks,patchsize)


# def align_burst_from_blocks_padded_patches(patches,blocks,nblocks,patchsize):
#     nimages,nsegs,nframes = patches.shape[:3]
#     c,h,w = patches.shape[3:]
#     ps = patchsize 
#     crops = np.zeros((nimages,nsegs,nframes,c,ps,ps)).astype(np.float)
#     ref_t = nframes//2
#     numba_align_burst_from_blocks_padded_patches(crops,patches,blocks,nblocks,patchsize)
#     return crops

# def align_burst_from_flow_global(bursts,flow,nblocks):
#     ref_blocks = global_flow_to_blocks(flow,nblocks)
#     t_blocks = global_blocks_ref_to_frames(ref_blocks,nblocks)
#     return align_burst_from_block_global(bursts,t_blocks,nblocks)

