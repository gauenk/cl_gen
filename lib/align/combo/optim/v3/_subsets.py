# -- python imports --
import nvtx
import numpy as np
import numpy.random as npr
from joblib import delayed
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- numba imports --
import numba
from numba import njit

# -- pytorch imports --
import torch
from torch.nn import functional as tnnF

# -- project imports --
from pyutils import torch_to_numpy
import align.combo._block_utils as block_utils
from patch_search import get_score_function

# -- [local] project imports --
from ._blocks import get_search_blocks

@nvtx.annotate("rand_subset_search", color="green")
def rand_subset_search(patches,masks,evaluator,curr_blocks,brange,nblocks,subsizes):
    r"""
    
    brange: a list of search ranges for each frame
    """

    # -- skip of subset sizes provided is empty --
    if len(subsizes) == 0: return torch_to_numpy(curr_blocks)
    
    # -- this would be a good place for particle filtering --
    # -- 1.) for each particle we would search along random subsets --
    # -- 2.) merge results periodically (e.g. take best one so far) --
    # -- 3.) alternate between (1) and (2)

    device = patches.device
    nimages,nsegs,nframes = patches.shape[:3]
    brange = torch_to_numpy(brange)
    curr_blocks = torch_to_numpy(curr_blocks)
    

    for size in subsizes:

        ps = evaluator.patchsize
        # frames = propose_frames(patches,curr_blocks,nframes,ps,size,nblocks)
        rands = npr.choice(nframes,size=size,replace=False)
        frames = repeat(rands,'z -> i s z',i=nimages,s=nsegs)
        srch_blocks = get_search_blocks(frames,brange,curr_blocks,device)
        scores,scores_t,blocks = evaluator.compute_topK(patches,masks,
                                                        srch_blocks,nblocks,1)
        blocks = torch_to_numpy(blocks)
        curr_blocks = blocks[:,:,0,:]
        torch.cuda.empty_cache()

    return curr_blocks


def propose_frames(patches,curr_blocks,nframes,ps,size,nblocks):
    if size == nframes:
        return propose_frames_rand(nframes,size)
    else:
        return propose_frames_ransac(patches,curr_blocks,ps,size,nblocks)

def propose_frames_ransac(patches,curr_blocks,ps,size,nblocks):
    score_fxn = get_score_function("bootstrapping_mod2")

    nimages,nsegs,nframes = patches.shape[:3]
    pcolor,ps_pad,ps_pad = patches.shape[3:]
    device = patches.device
    block_patches = np.zeros((nimages,nsegs,nframes,1,pcolor,ps,ps))
    block_patches = torch.FloatTensor(block_patches).to(device,non_blocking=True)
    curr_blocks = repeat(torch.LongTensor(curr_blocks).to(device),'i s t -> i s 1 t')
    tokeep = torch.IntTensor(np.arange(1)).to(device,non_blocking=True)

    # block_patches_nba = numba.cuda.as_cuda_array(block_patches)
    # patches_nba = numba.cuda.as_cuda_array(patches)
    # batch_nba = numba.cuda.as_cuda_array(curr_blocks)
    # block_utils.index_block_batches(block_patches_nba,
    #                                 patches_nba,batch_nba,
    #                                 ps,nblocks)
    block_patches_i = block_utils.index_block_batches(block_patches,patches,
                                                      curr_blocks,tokeep,
                                                      ps,nblocks)
    def compute_score_fxn(patches):
        nimages = patches.shape[0]
        patches = rearrange(patches,'b p t a c h w -> 1 (b p) a t c h w')
        cfg = edict({'gpuid':0})
        scores,scores_t = score_fxn(cfg,patches)
        scores = rearrange(scores,'1 (b p) a -> b p a',b=nimages)
        scores_t = rearrange(scores_t,'1 (b p) a t -> b p a t',b=nimages)
        scores_t = scores_t[:,:,0,:]
        return scores_t
    scores_t = compute_score_fxn(block_patches_i)
    scores_t = tnnF.normalize(scores_t,dim=2,p=1.).cpu().numpy()
    picked = np.zeros((nimages,nsegs,size))
    numba_choice_sample_mat(picked,scores_t,size)
    picked = picked.astype(np.int)
    return picked

# @njit
def numba_choice_sample_mat(picked,scores_t,size):
    nimages,nsegs,nframes = scores_t.shape
    for i in range(nimages):
        for s in range(nsegs):        
            frames = np.random.choice(nframes,size=size,p=scores_t[i,s],replace=False)
            picked[i,s,:] = frames
    return picked

def propose_frames_rand(nframes,size):
    rands = npr.choice(nframes,size=size,replace=False)
    frames = repeat(rands,'z -> i s z',i=nimages,s=nsegs)
    return frames
