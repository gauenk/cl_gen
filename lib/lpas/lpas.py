"""
Local patch alignment search

"""

from .utils import get_patches
from .optimizer import AlignmentOptimizer

def lpas_search(cfg,burst,motion=None):
    """
    cfg: edict with parameters
    burst: shape: B,T,C,H,W (B = batch, T = number of frames)
    """

    # -- def vars + create patches --
    B,T,C,H,W = burst.shape
    patches = get_patches(burst)
    B,R,T,H,C,H,W = patches.shape

    # -- create helper objects --
    optim = AlignmentOptimizer(patches,'global',1000,motion)
    
    # -- run optimizer --
    motion = optim.motion_sampler.init_sample()
    search_frames = optim.frame_sampler.init_sample()
    for i in range(optim.nsteps):
        fixed_frames = optim.frame_sampler.fixed(search_frames)
        align_scores = optim.block_sampler.sample(search_frames,fixed_frames,motion)
        motion = optim.motion_sampler.sample(align_scores)
        search_frames = optim.frame_sampler.sample(align_scores,motion)

    # -- return best ones from subsamples --
    nh_samples = optim.get_best_samples()
    
    # -- optimize over the final sets --
    (score,indices) = nh_samples[0]

    # -- recover aligned burst images --
    aligned = aligned_burst_image_from_indices_global_dynamics(patches,np.arange(T),indices)

    return score,aligned # indices for each neighborhood
        


