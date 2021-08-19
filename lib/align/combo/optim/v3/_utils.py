import torch
import numpy as np
from einops import repeat,rearrange

def get_boot_hyperparams(nframes,nblocks):
    if nframes == 3:
        K = nblocks**2
        subsizes = [nframes]
    elif nframes == 5:
        K = 2*nblocks
        subsizes = [nframes]
    elif nframes <= 20:
        K = 2*nblocks
        subsizes = [2,]*nframes
    else:
        K = nblocks
        subsizes = [2,]*nframes
    return K,subsizes

def get_ref_block(nblocks):
    return nblocks**2//2 + (nblocks//2)*(nblocks%2==0)

def pick_top1_blocks(topK_blocks):
    r"""
    Pick the best alignment from each frame's topK choices

    curr_blocks.shape = ( nimages, nsegs, nframes )
    topK_blocks = (nimages, nsegs, nframes, K)
    """
    curr_blocks = topK_blocks[:,:,:,0]
    return curr_blocks
    # curr_blocks = []
    # nframes = len(topK_blocks)
    # for t in range(nframes):
    #     curr_blocks.append(topK_blocks[t][:,:,0])
    # curr_blocks = torch.stack(curr_blocks,dim=-1)
    # return curr_blocks

def init_optim_block(nimages,nsegs,nframes,nblocks):
    ref_block = get_ref_block(nblocks)
    blocks = torch.ones((nimages,nsegs,nframes)).type(torch.long)
    blocks *= ref_block
    return blocks

def exh_block_range(nimages,nsegs,nframes,nblocks):
    full_range = torch.LongTensor([np.arange(nblocks**2) for t in range(nframes)])
    full_range = repeat(full_range,'t h -> b s t h',b=nimages,s=nsegs)
    full_range = full_range.type(torch.long)
    return full_range


