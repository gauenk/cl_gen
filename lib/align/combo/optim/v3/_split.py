
# -- python imports --
import copy,nvtx
import numpy as np
from joblib import delayed

# -- pytorch imports --
import torch

# -- project imports --
from pyutils import torch_to_numpy
from align._parallel import ProgressParallel

# -- [local] project imports --
from ._utils import get_ref_block
from ._blocks import  get_search_blocks

def init_topK_split_search(nimages,nsegs,nframes,K):
    topK_blocks = torch.zeros((nimages,nsegs,nframes,K)).type(torch.long)
    return topK_blocks

def split_frame_search(patches,masks,evaluator,curr_blocks,brange,nblocks,K):

    PARALLEL = False
    if PARALLEL:
        return split_frame_search_parallel(patches,masks,evaluator,
                                           curr_blocks,brange,nblocks,K)
    else:
        return split_frame_search_serial(patches,masks,evaluator,
                                         curr_blocks,brange,nblocks,K)
        
@nvtx.annotate("split_frame_search", color="blue")
def split_frame_search_parallel(patches,masks,evaluator,curr_blocks,brange,nblocks,K):

    # -- to numpy for meshgrid --
    device = patches.device
    patches = torch_to_numpy(patches)
    brange = torch_to_numpy(brange)
    curr_blocks = torch_to_numpy(curr_blocks)

    # -- shapes and init --
    nimages,nsegs,nframes = patches.shape[:3]
    ones = np.ones((nimages,nsegs,1))
    topK_blocks = init_topK_split_search(nimages,nsegs,nframes,K)
    assert nimages == 1,"Only batchsize 1 right now."

    # -- run split frame in parallel --
    args = [device,patches,masks,evaluator,ones,brange,curr_blocks,nblocks,K,topK_blocks]
    pParallel = ProgressParallel(use_tqdm=False,total=nframes,n_jobs=4)
    delayed_fxn = delayed(split_frame_search_single)
    pParallel(delayed_fxn(t,*args) for t in range(nframes))

    # blocks.shape = nimages, nsegs, nframes, K
    return topK_blocks

def split_frame_search_single(t,device,patches,masks,evaluator,ones,brange,
                              curr_blocks,nblocks,K,topK_blocks):

    # -- to numpy for meshgrid --
    patches = torch.FloatTensor(patches).to(device,non_blocking=True)
    evaluator = copy.deepcopy(evaluator)
    nframes = topK_blocks.shape[2]
    if t == nframes//2:
        ref_block = get_ref_block(nblocks)
        topK_blocks[:,:,t,:] = ref_block
    else:
        blocks_t = ones * t
        srch_blocks = mesh_block_ranges(blocks_t,brange,curr_blocks)
        srch_blocks = torch.LongTensor(srch_blocks).to(device,non_blocking=True)
        scores,blocks = evaluator.compute_topK_scores(patches,masks,srch_blocks,
                                                      nblocks,K)
        # blocks.shape = nimages, nsegs, K, nframes
        topK_blocks[:,:,t,:] = blocks[:,:,:,t]

def split_frame_search_serial(patches,masks,evaluator,curr_blocks,brange,nblocks,K):
    r"""
    
    brange: a list of search ranges for each frame
    """
    # -- shapes and init --
    nimages,nsegs,nframes = patches.shape[:3]
    ones = np.ones((nimages,nsegs,1))
    topK_blocks = init_topK_split_search(nimages,nsegs,nframes,K)
    assert nimages == 1,"Only batchsize 1 right now."
    ref_block = get_ref_block(nblocks)
    device = patches.device

    brange = torch_to_numpy(brange)
    curr_blocks = torch_to_numpy(curr_blocks)

    for t in range(nframes):
        if t == nframes//2:
            topK_blocks[:,:,t,:] = ref_block
            continue
        blocks_t = ones * t
        bsize = evaluator.block_batchsize
        srch_blocks = get_search_blocks(blocks_t,brange,curr_blocks,device,False)
        # srch_blocks = mesh_block_ranges_gen(blocks_t,brange,curr_blocks,bsize,device)
        # srch_blocks = mesh_block_ranges(blocks_t,brange,curr_blocks,device)
        scores,blocks = evaluator.compute_topK_scores(patches,masks,srch_blocks,
                                                        nblocks,K)
        # blocks.shape = nimages, nsegs, K, nframes
        topK_blocks_t = blocks[:,:,:,t]
        topK_blocks[:,:,t,:] = topK_blocks_t
        torch.cuda.empty_cache()


    return topK_blocks

