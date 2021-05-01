"""
Local patch alignment search

"""

# -- python imports --
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --

# -- [local] project imports --
from .utils import get_sobel_patches,save_image,get_ref_block_index,delta_to_block,align_burst_from_block
from .optimizer import AlignmentOptimizer

def lpas_search(cfg,burst,motion=None):
    """
    cfg: edict with parameters
    burst: shape: B,T,C,H,W (B = batch, T = number of frames)
    """
    if not(motion is None):
        print(delta_to_block(motion,cfg.nblocks))

    # -- def vars + create patches --
    nframes,nblocks = cfg.nframes,cfg.nblocks
    T,B,C,H,W = burst.shape 
    num_patches,patchsize = 10,32
    patches,locations = get_sobel_patches(burst,nblocks,num_patches,patchsize)
    R,B,T,H,C,PS,PS = patches.shape

    # -- create helper objects --
    nsteps = 1
    nparticles = 100
    isize = burst.shape[-1]**2
    score_params = edict({'stype':'raw','name':'ave'})
    optim = AlignmentOptimizer(nframes,nblocks,isize,'global_const',
                               nsteps,nparticles,motion,score_params)
    
    # -- init states of optimizer --
    motion = optim.motion_sampler.init_sample()
    fixed_frames = optim.frame_sampler.init_sample()
    block_grids = torch.LongTensor([[nblocks**2//2 for t in range(nframes)]])
    scores,blocks = optim.sample(patches,block_grids)

    # -- search --
    for i in range(optim.nsteps):
        fixed_frames = optim.frame_sampler.sample()
        block_grids = optim.block_sampler.sample(fixed_frames,motion)
        scores,blocks = optim.sample(patches,block_grids)
        motion = optim.motion_sampler.sample(scores,motion)
        # search_frames = optim.frame_sampler.sample(scores,motion)

    # -- return best ones from subsamples --
    scores,blocks = optim.get_best_samples()
    
    # -- optimize over the final sets --
    score,block = scores[:,0],blocks[:,0]

    # -- recover aligned burst images --
    aligned = align_burst_from_block(cfg,burst,block,"global")

    return score,aligned # indices for each neighborhood
        


