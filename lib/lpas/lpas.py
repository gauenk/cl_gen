"""
Local patch alignment search

"""

from .utils import get_patches
from .optimizer import AlignmentOptimizer

def lpas(cfg,burst):
    """
    cfg: edict with parameters
    burst: shape: B,T,C,H,W (B = batch, T = number of frames)
    """

    # -- def vars + create patches --
    B,T,C,H,W = burst.shape
    patches = get_patches(burst)
    B,R,T,H,C,H,W = patches.shape

    # -- create helper objects --
    optim = AlignmentOptimizer(patches,'global',1000)
    
    # -- run optimizer --
    search_frames = optim.frame_sampler.init_sample()
    for i in range(optim.nsteps):
        fixed_frames = optim.frame_sampler.fixed(search_frames)
        samples = optim.block_sampler.sample(search_frames,fixed_frames)

    # -- return best ones from subsamples --
    nh_samples = optim.get_best_samples()
    
    # -- optimize over the final sets --
        
    return nh_indices # indices for each neighborhood
        


