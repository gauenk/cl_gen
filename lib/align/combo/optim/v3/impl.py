
# -- python imports --
import copy
import numpy as np
import numpy.random as npr
from einops import rearrange,repeat
from joblib import delayed

# -- pytorch imports --
import torch

# -- numba imports --
from numba import jit,prange

# -- project imports --
from patch_search import get_score_function
from align.xforms import blocks_to_flow
from align._parallel import ProgressParallel
from align.combo import EvalBlockScores
    
# -- [local] project imports --
from ._utils import get_ref_block,init_optim_block,exh_block_range,pick_top1_blocks
from ._split import split_frame_search
from ._subsets import rand_subset_search

def run(patches,masks,evaluator,
        nblocks,iterations,
        subsizes,K,proc_idx):

    # TODO: if patches are not rectangles, create masks and *USE*
    # -- currently they dont do anything.
    # Current: assume patches are all rectangles
    # nftrs is flattened from ( C x max_seg_H x max_seg_W )
    # "mask" says where padding exists, ( C x max_seg_H x max_seg_W )
    # "locs" with shape (nimages,nframes,nsegs,nftrs,2) says where each ftr came from
    # actual features is flattened from ( C x max_seg_H x max_seg_W ) "and-ed" with a mask

    # -- init settings --
    evaluator = copy.deepcopy(evaluator)
    gpuid = gpuid_from_proc(proc_idx)
    evaluator.gpuid = gpuid
    nimages,nsegs,nframes = patches.shape[:3]
    assert nimages == 1,"Single batch per search."
    curr_blocks = init_optim_block(nimages,nsegs,nframes,nblocks)
    exh_brange = exh_block_range(nimages,nsegs,nframes,nblocks)

    # -- swap device --
    patches = patches.to(f'cuda:{gpuid}',non_blocking=True)
    masks = masks.to(f'cuda:{gpuid}',non_blocking=True)
    
    # -- debug --
    frame_size = int(np.sqrt(nsegs))
    is_even = frame_size%2 == 0
    mid_edge = frame_size*frame_size//2
    mid_pix = frame_size*frame_size//2 + (frame_size//2)*is_even
    mid_pix = 32*10 + 23

    # -- initial search using "ave" --
    ave_eval = create_ave_eval(evaluator)
    # curr_blocks = init_split_search(patches,masks,ave_eval,
    #                                 curr_blocks,exh_brange,nblocks)
    topK_blocks = split_frame_search(patches,masks,ave_eval,curr_blocks,
                                     exh_brange,nblocks,K)
    curr_blocks = pick_top1_blocks(topK_blocks)



    # -- continued search using "search_fxn" --
    for iter_i in range(iterations):

        # -- pick topK arrangements searching each frame separately --
        # topK_blocks = split_frame_search(patches,masks,evaluator,curr_blocks,
        #                                  exh_brange,nblocks,K)
        # topK_blocks = split_frame_search(patches,masks,ave_eval,curr_blocks,
        #                                  exh_brange,nblocks,K)
        # curr_blocks = pick_top1_blocks(topK_blocks)
        
        # -- pick top arrangement search over rand subsets of frames --
        curr_blocks = rand_subset_search(patches,masks,evaluator,curr_blocks,
                                         topK_blocks,nblocks,subsizes)
    # if iterations > 1:
    #     print("WARNING: Iterations > 1 have no meaning.")
    flow = blocks_to_flow(curr_blocks,nblocks) # 'i s t two'
    return flow

def create_ave_eval(evaluator):
    score_fxn = get_score_function("ave")
    ave_eval = EvalBlockScores(score_fxn,"ave",evaluator.patchsize,
                                evaluator.block_batchsize,
                                evaluator.noise_info,evaluator.gpuid)
    return ave_eval

def init_split_search(patches,masks,ave_eval,curr_blocks,exh_brange,nblocks):
    topK_blocks = split_frame_search(patches,masks,ave_eval,curr_blocks,
                                     exh_brange,nblocks,1)
    curr_blocks = pick_top1_blocks(topK_blocks)
    return curr_blocks
    
def gpuid_from_proc(proc_idx):
    mod = proc_idx % 3 
    if mod in [0,1]:
        gpuid = 1
    else:
        gpuid = 2
    # return gpuid
    return 1

"""

(A) list of frames, single block, nblocks range for each frame
    - for a fixed block arangement, compute all permutations along a frame dim
(B) list of frames, a list of blocks, nblocks range for each frame
    - for a set of block arangements (one per frame)


K^S_n search over the top_K for each S_n frame randomly sampled

In (A) the range of each t is { 1, ..., nblocks }
In (B) the range of each t is { k_1, ..., k_K }

We need a fixed value for each frame who is not selected. ("optim_block")

We compute meshgrid over the selected frame's range

"""
