"""
Local patch alignment search

"""

# -- python imports --
import numpy.random as npr
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --

# -- [local] project imports --
from pyutils import align_burst_from_flow,align_burst_from_block,global_flow_to_blocks,global_blocks_to_flow,print_tensor_stats
from .utils import get_sobel_patches,save_image,get_ref_block_index
from .optimizer import AlignmentOptimizer

def lpas_spoof(burst,motion,nblocks,mtype,acc):
    T = burst.shape[0]
    ref_block = get_ref_block_index(nblocks)
    gt_blocks = global_flow_to_blocks(motion,nblocks)
    rands = npr.uniform(0,1,size=motion.shape[0])
    scores,blocks = [],[]
    for idx,rand in enumerate(rands):
        if rand > acc:
            fake = torch.randint(0,nblocks**2,(T,))
            fake[T//2] = ref_block
            blocks.append(fake)
        else: blocks.append(gt_blocks[idx])
        scores.append(0)
    blocks = torch.stack(blocks)
    burst_clone = burst.clone()
    aligned = align_burst_from_block(burst_clone,blocks,nblocks,"global")
    # print_tensor_stats("[lpas]: burst0 - burst1",burst[0] - burst[1])
    # print_tensor_stats("[lpas]: aligned0 - aligned1",aligned[0] - aligned[1])
    return scores,aligned

def lpas_search(cfg,burst,motion=None):
    """
    cfg: edict with parameters
    burst: shape: T,B,C,H,W (T = number of frames, B = batch)
    """
    # if not(motion is None):
    # gt_motion = global_flow_to_blocks(motion,cfg.nblocks)
    # flow = global_blocks_to_flow(gt_motion,cfg.nblocks)

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
    
    # -- execute a simple search (one with many issues and few features) --
    simple_search(optim,patches)

    # -- return best ones from subsamples --
    scores,blocks = optim.get_best_samples()
    
    # -- optimize over the final sets --
    score,block = scores[:,0],blocks[:,0]

    # -- dynamic error --
    dynamic_acc = 0
    for b in range(B):
        dynamic_acc += torch.all(gt_motion[b].cpu() == block[b].cpu())
    # print(score)
    # print(dynamic_acc/B)

    # -- recover aligned burst images --
    aligned = align_burst_from_block(burst,block,cfg.nblocks,"global")

    return score,aligned # indices for each neighborhood



def simple_search(optim,patches):

    # -- init states of optimizer --
    motion = optim.motion_sampler.init_sample()
    fixed_frames = optim.frame_sampler.init_sample()
    block_grids = optim.block_sampler.init_sample()
    scores,blocks = optim.sample(patches,block_grids)

    # -- search --
    for i in range(optim.nsteps):

        """
        1. init samples
        2. seach each frame separately
        3. keep top k
        4. search meshgrid of k^(t-1)
        5. pick minima
        """

        # -- search along each separate frame --
        fixed_frames = optim.frame_sampler.sample()
        block_grids = optim.block_sampler.sample(fixed_frames,motion,"split")
        scores,blocks = optim.sample(patches,block_grids)
        motion = optim.motion_sampler.sample(scores,motion)

        # -- search along frame mesh frame --
        block_grids = optim.block_sampler.sample(fixed_frames,motion,"mesh")


        # search_frames = optim.frame_sampler.sample(scores,motion)
        
