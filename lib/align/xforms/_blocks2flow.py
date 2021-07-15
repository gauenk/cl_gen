# -- python imports --
import numpy as np
from einops import rearrange

# -- numba imports --
from numba import jit,prange
from numba.extending import overload

# -- pytorch imports --
import torch

# -- project imports --
import align.xforms._utils as utils
from align._utils import torch_to_numpy

def global_blocks_to_flow(_flow,nblocks):
    print("[DEPRECATED: global_blocks_to_flow] use [blocks_to_flow_serial]")
    return blocks_to_flow_serial(_flow,nblocks)

def blocks_to_flow_serial(blocks,nblocks):
    r"""
    flow 
    """
    B,T = blocks.shape[:2]
    grid = np.arange(nblocks**2).reshape(nblocks,nblocks)
    flow = []
    ref_t,ref_bl = T//2,nblocks//2
    for b in range(B):
        block_b = blocks[b]
        coords = []
        for t in range(T):
            coord = np.r_[np.where(grid == block_b[t].item())]
            coords.append(coord)
        coords = np.stack(coords)

        # -- x <-> y swap -- rows are "y" and cols are "x" -- want (x,y)
        coords = coords[:,::-1] 

        # -- Top-Left_Matrix_coordinates -> Obj_Spatial_coordinates --
        coords[:,0] *= -1 

        # -- compute the flow --
        coords[:,0] = np.ediff1d(coords[:,0],0)
        coords[:,1] = np.ediff1d(coords[:,1],0)

        # -- remove the last one -- it's value is [0,0] --
        flow_b = coords[:-1] 

        flow.append(flow_b)

    flow = np.stack(flow)
    flow = torch.LongTensor(flow)
    return flow

def blocks_to_flow(blocks,nblocks):

    # -- required dims --
    nimages,npix,nframes = blocks.shape

    # -- compute conversion --
    blocks = torch_to_numpy(blocks)
    blocks = rearrange(blocks,'i p t -> (i p) t')
    flow = blocks_to_flow_numba(blocks,nblocks)
    flow = rearrange(flow,'(i p) tm1 two -> i p tm1 two',i=nimages)

    # -- back to torch --
    flow = torch.LongTensor(flow)

    return flow

@jit(nopython=True)
def blocks_to_flow_numba(blocks,nblocks):
    r"""
    flow 
    """
    B,T = blocks.shape[:2]
    grid = np.arange(nblocks**2).reshape(nblocks,nblocks).astype(np.int64)
    flow = np.zeros((B,T-1,2))
    ref_t,ref_bl = T//2,nblocks//2
    for b in range(B):
        block_b = blocks[b]
        coords = np.zeros((T,2),dtype=np.int64)
        for t in range(T):

            # -- not numba friendly --
            # coord = np.r_[np.where(grid == block_b[t])]
            # coords[t,:] = coord

            # -- numba friendly --
            coords_t = np.where(grid == block_b[t])            
            coords[t,0] = coords_t[0][0]
            coords[t,1] = coords_t[1][0]

        # -- x <-> y swap -- rows are "y" and cols are "x" -- want (x,y)
        coords = coords[:,::-1] 

        # -- Top-Left_Matrix_coordinates -> Obj_Spatial_coordinates --
        coords[:,0] *= -1 

        # -- compute the flow --
        coords[:,0] = np.ediff1d(coords[:,0],0)
        coords[:,1] = np.ediff1d(coords[:,1],0)

        # -- remove the last one -- it's value is [0,0] --
        flow_b = coords[:-1] 

        flow[b] = flow_b

    # flow = np.stack(flow)
    # flow = torch.LongTensor(flow)
    return flow


