
# -- python imports --
import math
import numpy as np
from einops import rearrange

# -- numba imports --
from numba import jit,prange


# -- pytorch imports --
import torch


def blocks_shape(blocks):
    nimage = len(blocks)
    nsegs = len(blocks[0])
    naligns = len(blocks[0][0])
    nframes = len(blocks[0][0][0])
    return nimage,nsegs,naligns,nframes

def iter_block_batches(blocks,batchsize):
    nimage,nsegs,naligns,nframes = blocks.shape#blocks_shape(blocks)
    nbatches = naligns // batchsize
    nbatches += naligns % batchsize > 0
    for i in range(nbatches):
        start = i * batchsize
        end = start + batchsize
        batch = blocks[...,start:end,:]
        yield batch

    
# NOT CONCEPTUALLY TESTED;
# Are these patches the patches (tensor) we want?
# Connects "Tiling" with "Indexing" with "Blocks"
@jit(nopython=True)
def index_block_batches(indexed,tensor,batch,patchsize,nblocks):
    nimages,nsegs,nframes = tensor.shape[:3]
    color,pH_buf,pW_buf = tensor.shape[3:]
    nimages,nsegs,naligns,nframes = batch.shape
    ps = patchsize
    # indexed = np.zeros((nimages,nsegs,nframes,naligns,color,ps,ps))

    for i in prange(nimages):
        for s in prange(nsegs):
            for t in range(nframes):
                tensor_ist = tensor[i][s][t]
                for a in range(naligns):
                    bindex = batch[i][s][a][t]
                    # -- v1 (init) --
                    # hs = (bindex // nblocks)
                    # ws = (bindex % nblocks)

                    # -- v2 (object to camera) --
                    hs = nblocks-(bindex // nblocks) -1
                    ws = nblocks-(bindex % nblocks) - 1
                    # if s == 0:
                    #     print(t,a,bindex,hs,ws)
                    
                    # -- v3 (ref optical_flow code) --
                    # hs = (bindex // nblocks)
                    # ws = nblocks - (bindex % nblocks) - 1

                    # -- v4 (did i swap v3 somehow?) --
                    # hs = nblocks - (bindex // nblocks) - 1
                    # ws = (bindex % nblocks)

                    # if s == 0:
                    #     print(t,bindex,hs,ws)
                    he = hs + patchsize
                    we = ws + patchsize
                    indexed[i,s,t,a,...] = tensor_ist[:,hs:he,ws:we]
    return indexed

# def index_block_batches(tensor,batch,patchsize):

#     # -- get patches to tile --
#     nimages,nsegs,nframes = tensor.shape[:3]
#     color,pH,pW = tensor.shape[3:]
    
#     nimage,nsegs,nframes,batchsize = batch.shape


#     # -- option 1 -- ( small nsegs v.s. nblocks )
#     # iterate over all nimages,nsegs,nframes

#     # -- option 2 -- ( large nsegs v.s. nblocks )
#     # generate all nblocks and index using "gather"

#     # -- option 3 -- (numba?)
#     # iterate over all nimages,nsegs,nframes using numba and indexing

#     # batch = repeat(batch,'i t s b -> i t s b f',f=ftrs)

#     indexed = torch.gather(tensor,3,batch)


#     return indexed

def index_block_batches_deprecated(blocks,patches,indexing):
    assert len(patches.shape) == 7, "patches must include 7 dims"
    i = indexing
    ndims = len(blocks.shape)
    bbs = blocks.shape[1]
    block_patches = patches[i.bmesh[:,:bbs],:,i.tmesh[:,:bbs],blocks,:,:,:] 
    block_patches = rearrange(block_patches,'b e t r c ph pw -> b r e t c ph pw')
    return block_patches

def block_batch_update(samples,scores,blocks,K): # store as heap?

    # scores = torch.FloatTensor(samples.scores)
    # blocks = torch.LongTensor(samples.blocks)

    scores = torch.FloatTensor(scores)
    blocks = torch.LongTensor(blocks)

    if len(samples.scores) == 0:
        samples.scores = scores
        samples.blocks = blocks
    else:
        sscores = samples.scores
        sblocks = samples.blocks
        samples.scores = torch.cat([sscores,scores],dim=2)
        samples.blocks = torch.cat([sblocks,blocks],dim=2)
        
def get_block_batch_topK(samples,K):

    # -- get samples --
    scores = samples.scores
    blocks = samples.blocks

    # -- pick top K --
    nimages,nsegs,naligns = scores.shape
    if naligns < K: K = naligns
    scores_topK,blocks_topK = [],[]
    for i in range(nimages):
        for s in range(nsegs):
            topK = torch.topk(scores[i,s],K,largest=False)
            scores_topK.append(topK.values)
            blocks_topK.append(blocks[i,s,topK.indices])

    # -- stack em up! --
    scores_topK = torch.stack(scores_topK,dim=0)
    blocks_topK = torch.stack(blocks_topK,dim=0)

    # -- shape em up! --
    scores_topK = rearrange(scores_topK,'(b s) k -> b s k',b=nimages)
    blocks_topK = rearrange(blocks_topK,'(b s) k t -> b s k t',b=nimages)

    return scores_topK,blocks_topK
