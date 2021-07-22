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
from pyutils import torch_to_numpy


def global_flow_to_blocks(_flow,nblocks):
    print("[DEPRECATED: global_flow_to_blocks] use [flow_to_blocks_serial]")
    return flow_to_blocks_serial(_flow,nblocks)

def flow_to_blocks_serial(_flow,nblocks):
    r"""
    flow.shape = (Num of Samples, Num of Frames - 1, 2)

    "flow" is a integer direction of motion between two frames
       flow[b,t] is a vector of direction [dx, dy] wrt previous frame
    x goes left to right and y goes top to bottom

    b = image batch
    t = \delta frame index
    t_reference = T // 2

    "nblocks" is the maximum number of pixels changed between adj. frames
    "indices" are the 
       indices[b,t] is the integer index representing the specific
       neighbor for a given flow

    Returns:
    indices (ndarray): (nimages, nframes)
    """
    flow = _flow.clone()
    B,Tm1 = flow.shape[:2]
    T = Tm1 + 1
    grid = np.arange(nblocks**2).reshape(nblocks,nblocks)
    ref_t,ref_bl = T//2,nblocks//2
    indices,coord_ref = [],torch.IntTensor([[ref_bl,ref_bl]])
    for b in range(B):
        flow_b = flow[b]
        flow_b[:,0] *= -1 # -- spatial dir. to image coordinates (dx flips, dy same) --
        left = ref_bl - rcumsum(flow_b[:ref_t]) # -- moving backward
        right = torch.cumsum(flow_b[ref_t:],0) + ref_bl # -- moving forward
        coords = torch.cat([left,coord_ref,right],dim=0)
        assert np.all(coords >= 0), "all coordinates are non-negative."
        for t in range(T):
            x,y = coords[t][0].item(),coords[t][1].item()
            index = grid[y,x] # up&down == rows, left&right == cols
            indices.append(index)
    indices = torch.LongTensor(indices)
    indices = rearrange(indices,'(b t) -> b t',b=B)
    return indices

def flow_to_blocks(flow,nblocks):

    # -- check shapes --
    nimages,npix,nframes_minus_1,two = flow.shape

    # -- ensure int64 ndarray --
    flow = torch_to_numpy(flow)
    flow = flow.astype(np.int64)

    # -- create blocks --
    flow = rearrange(flow,'i p tm1 two -> (i p) tm1 two')
    blocks = flow_to_blocks_numba(flow,nblocks)
    blocks = rearrange(blocks,'(i p) t -> i p t',i=nimages)

    # -- to tensor --
    blocks = torch.LongTensor(blocks)

    return blocks

@overload(np.flipud)
def np_flip_ud(a):
    def impl(a):
        return a[::-1, ...]
    return impl    

@jit(nopython=True)
def flow_to_blocks_numba(_flow,nblocks):
    r"""
    flow.shape = (nsamples,nframes-1,2)

    "flow" is a integer direction of motion between two frames
       flow[b,t] is a vector of direction [dx, dy] wrt previous frame
    x goes left to right and y goes top to bottom

    b = image batch
    t = \delta frame index
    t_reference = T // 2

    "nblocks" is the maximum number of pixels changed between adj. frames
    "indices" are the 
       indices[b,t] is the integer index representing the specific
       neighbor for a given flow

    Returns:
    indices (ndarray): (nimages, nframes)
    """

    def np_2dcumsum(array2d):
        x = array2d[:,0].cumsum()
        y = array2d[:,1].cumsum()
        stack = np.stack((x,y)).T
        return stack

    def np_rcumsum(tensor,dim=0):
        return np.flipud(np_2dcumsum(np.flipud(tensor)))


    flow = _flow.copy()
    B,Tm1 = flow.shape[:2]
    T = Tm1 + 1
    grid = np.arange(nblocks**2).reshape(nblocks,nblocks).astype(np.int64)
    ref_t,ref_bl = T//2,nblocks//2
    coord_ref = np.array([[ref_bl,ref_bl]],dtype=np.int64)

    indices = np.zeros((B,T),dtype=np.int64)
    for b in prange(B):
        flow_b = flow[b]
        # convert the 
        #       _spatial_  _object_ motion
        # into
        #      _image coordinate_ of _top-left frame_ 
        # (dx flips, dy same) <==> (dx flips for _object_ to _top-left_) (dy flips twice)
        flow_b[:,0] *= -1 
        left = ref_bl - np_rcumsum(flow_b[:ref_t]) # -- moving backward
        right = np_2dcumsum(flow_b[ref_t:]) + ref_bl # -- moving forward
        left = left.reshape(-1,2)
        right = right.reshape(-1,2)
        coords = np.concatenate((left,coord_ref,right))
        assert np.all(coords >= 0), "all coordinates are non-negative."
        for t in range(T):
            x,y = int(coords[t][0]),int(coords[t][1])
            index = grid[y,x] # up&down == rows, left&right == cols
            indices[b,t] = index
    return indices

