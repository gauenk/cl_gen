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
    nframes = burst.shape[0]
    pix = flow_to_pix(flow,nframes,centers,isize)
    return align_from_pix(burst,pix,nblocks)

def align_from_blocks(burst,blocks,nblocks,centers=None,isize=None):
    pix = blocks_to_pix(blocks,nblocks,centers,isize)
    return align_from_pix(burst,pix,nblocks)

# -=-=-=-=-=-=-=-=-=-=-=-
#
#  -- Align from Pix --
#
# -=-=-=-=-=-=-=-=-=-=-=-

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
    pad_burst = torch_to_numpy(pad_burst).astype(np.float64)
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
    pad = nblocks//2
    h = h_pad - 2*pad
    w = w_pad - 2*pad
    npix = h*w
    t_ref = nframes//2
    for i in range(nimages):
        for t in range(nframes):
            for p in range(npix):
                # ref_xy =  pix[i,p,t_ref] #ordering of "p" is not always order of pix.
                # r_col,r_row = ref_xy[0],ref_xy[1]

                r_row,r_col = p//w,p%w
                xy_xfer = pix[i,p,t]
                b_col,b_row = xy_xfer[0],xy_xfer[1]
                # if 15 < r_row and r_row < 18 and 15 < r_col and r_col < 18:
                #     print(r_row,r_col,b_row,b_col)
                # if 30 < r_row and r_row < 32 and 30 < r_col and r_col < 32:
                #     print(r_row,r_col,b_row,b_col)

                b_row,b_col = b_row+pad,b_col+pad
                # r_row,r_col = r_row+pad,r_col+pad
                # if b_row < 0 or b_col < 0 or b_row > 31 or b_col > 31:
                #     print("oob b",b_row-pad,b_col-pad)
                # aligned[t,i,:,r_row,r_col] = burst[t,i,:,b_row,b_col]
                aligned[t,i,:,r_row,r_col] = burst[t,i,:,b_row,b_col]
                # aligned[t,i,:,b_row,b_col] = burst[t,i,:,r_row+pad,r_col+pad]


                # aligned[t,i,:,r_row,r_col] = burst[t_ref,i,:,r_row+pad,r_col+pad]

                # -- old interpretation --
                # aligned[t,i,:,b_row,b_col] = burst[t,i,:,r_row,r_col]
                # burst[t,i,:,r_row,r_col] \sim burst[t_ref,i,:,b_row,b_col]
                # WANT:
                # aligned[t,i,:,r_row,r_col] \sim burst[t_ref,i,:,r_row,r_col]

                # how do we move "r_row" to "b_row"?

                # r_row,r_col = p//w,p%w
                # xy_xfer = pix[i,p,t]
                # b_col,b_row = xy_xfer[0],xy_xfer[1]
                # r_row,r_col = r_row+pad,r_col+pad
                # aligned[t,i,:,b_row,b_col] = burst[t,i,:,r_row,r_col]


                #
                # -- this code recovers the clean, dynamic motion --
                #

                # r_row,r_col = p//w,p%w
                # xy_xfer = pix[i,p,t]
                # b_col,b_row = xy_xfer[0],xy_xfer[1]
                # b_row,b_col = b_row+pad,b_col+pad
                # aligned[t,i,:,r_row,r_col] = burst[t_ref,i,:,b_row,b_col]


# -=-=-=-=-=-=-=-=-=-=-=-
#
#  -- Align from Flow --
#
# -=-=-=-=-=-=-=-=-=-=-=-

def align_from_flow_notimpl(burst,flow,nblocks):
    r"""
    Align a burst of frames 
    from a pixel tensor representing 
    each pixel's adjustment
    """

    # -- shapes --
    nframes,nimages,c,h,w = burst.shape
    nimages,npix,nframes_m1,two = flow.shape
    nframes = nframes_m1 + 1

    # -- add paddings --
    burst = rearrange(burst,'t i c h w -> (t i) c h w')
    pad_burst = F.pad(burst,[nblocks//2,]*4,mode='reflect')
    pad_burst = rearrange(pad_burst,'(t i) c h w -> t i c h w',i=nimages)

    # -- ensure ndarray --
    pad_burst = torch_to_numpy(pad_burst)
    flow = torch_to_numpy(flow)

    # -- create blocks --
    aligned = np.zeros((nframes,nimages,c,h,w)).astype(np.float64)
    pad_burst = pad_burst.astype(np.float64)
    align_from_flow_numba(aligned,pad_burst,flow,nblocks)

    # -- back to torch --
    aligned = torch.FloatTensor(aligned)
    aligned[nframes//2] = burst[nframes//2].clone()

    return aligned

@jit(nopython=True)
def align_from_flow_numba(aligned,burst,flow,nblocks):
    nframes,nimages,c,h_pad,w_pad = burst.shape
    pad = nblocks//2
    h = h_pad - 2*pad
    w = w_pad - 2*pad
    npix = h*w
    t_ref = nframes//2
    for i in range(nimages):
        for t_i,t in enumerate(range(nframes)):
            if t == t_ref: continue
            if t_i > t_ref: t_i -= 1
            for p in range(npix):
                r_row,r_col = p//w,p%w

                deltas = flow[i,p,t_i]
                delta_x,delta_y = deltas[0],deltas[1]

                b_row = r_row + delta_x
                b_col = r_col + delta_y
                # print(delta_x,delta_y,r_row,r_col,b_row,b_col)

                aligned[t,i,:,r_row,r_col] = burst[t_ref,i,:,b_row,b_col]


