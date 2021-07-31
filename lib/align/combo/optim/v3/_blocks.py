# -- python imports --
import numba
import torch
import numpy as np
from einops import rearrange,repeat

# -- pytorch imports --
from torch.nn.utils.rnn import pad_sequence

# -- numba imports --
from numba import jit,prange

# -- project imports --
from pyutils.mesh_gen import gen_indexing_mesh_levels,BatchGen

MAX_FRAMES = 15

def get_search_blocks(blocks_t,brange,curr_blocks,device=None,permit_gen=True):
    nframes = len(brange[0][0])
    if nframes >= MAX_FRAMES and permit_gen:
        return mesh_block_ranges_gen(blocks_t,brange,curr_blocks,device=device)
    else:
        return mesh_block_ranges(blocks_t,brange,curr_blocks,device=device)

def mesh_block_ranges_gen(blocks_t,brange,curr_blocks,device=None):
    block_ranges = select_block_ranges(blocks_t,brange,curr_blocks)
    srch_blocks = mesh_blocks_gen(block_ranges,device)
    return srch_blocks
    
def mesh_block_ranges(blocks_t,brange,curr_blocks,device=None):
    block_ranges = select_block_ranges(blocks_t,brange,curr_blocks)
    srch_blocks = mesh_blocks(block_ranges,device)
    return srch_blocks

def select_block_ranges(frames,brange,curr_blocks):
    r"""
    frames: [ frame_index_1, frame_index_2, ..., frame_index_F ]
    brange[0,:,0]: [ [range_of_frame_1], [range_of_frame_2], ..., [range_of_frame_T] ]
    curr_blocks: [ block_1, block_2, ..., block_T ]

    With "F" (num of selected frames) < "T" (num of total frames)

    frames.shape = (nimages,nsegs,M)
    brange.shape = (nimages,nsegs,nframes,K) for K = Num of Possible Values

    create list of ranges (lists) for each frames for a meshgrid
    
    sranges[nested lists] shape = (nimages,nsegs,nframes,*)
    """

    def select_block_ranges_bs(frames,brange,curr_blocks):
        srange,nframes = [],len(curr_blocks)
        for f in range(nframes):
            if f in frames:
                brange_u = np.unique(brange[f])
                selected_indices = np.atleast_1d(brange_u)
            else:
                selected_indices = np.atleast_1d(curr_blocks[f])
            srange.append(list(selected_indices))
        return srange
    
    nimages = len(brange)
    nsegs = len(brange[0])
    nframes = len(brange[0][0])
    sranges = []
    # sranges = [[[] for j in range(nsegs)] for i in range(nimages)]
    for b in range(nimages):
        sranges_b = []
        for s in range(nsegs):
            srange_bs = select_block_ranges_bs(frames[b][s],
                                               brange[b][s],
                                               curr_blocks[b][s])
            # sranges[b][s] = srange_bs
            sranges_b.append(srange_bs)
        sranges.append(sranges_b)
    return sranges

def mesh_blocks(brange,device):
    r"""

    "brange.shape" = (nimages, nsegs, nframes, K_f)
    with K_f different for each frame f.
    
    "mesh.shape" = (nimages, nsegs, naligns, nframes)
    """
    mesh = []
    nimages,nsegs = len(brange),len(brange[0])
    for b in range(nimages):
        mesh_b = []
        for s in range(nsegs):
            brange_bs = brange[b][s]
            grids = np.meshgrid(*brange_bs,indexing='ij')
            grids = [grids[g].flatten() for g in range(len(grids))]
            grids = rearrange(np.stack(grids),'t a -> a t')
            grids = torch.LongTensor(grids).to(device,non_blocking=True)
            mesh_b.append(grids)
        mesh_b = pad_sequence(mesh_b,batch_first=True,padding_value=0)
        # mesh_b = torch.stack(mesh_b,dim=0)
        mesh.append(mesh_b)
    mesh = torch.stack(mesh,dim=0)
    # mesh = rearrange(mesh,'(b s) t a -> b s a t',b=nimages)
    return mesh

def mesh_blocks_gen(brange,device):
    r"""
    Generator over the meshgrid.
    """

    def create_nested_gens(brange,levels_K,levels_H,device):
        nimages,nsegs = len(brange),len(brange[0])
        gens = []
        for i in range(nimages):
            gens_i = []
            for s in range(nsegs):
                range_is = brange[i][s]
                generator = gen_indexing_mesh_levels(range_is,levels_K,levels_H,device)
                gens_i.append(generator)
            gens.append(gens_i)
        return gens

    nframes = len(brange[0][0])
    if nframes < MAX_FRAMES:
        raise ValueError("Too few frames for generator to be efficient.")
    levels_K,levels_H = [3],[3]
    gens = create_nested_gens(brange,levels_K,levels_H,device)
    nframes = len(brange[0][0])
    max_range = np.max([len(brange[0][0][i]) for i in range(nframes)])
    batch_gens = BatchGen(gens,nframes,max_range,device)
    return batch_gens

