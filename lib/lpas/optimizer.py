
import numpy as np
import torch

from .samplers import FrameIndexSampler,BlockIndexSampler

class AlignmentOptimizer():

    def __init__(self,patches,motion_type,nsteps):
        self.patches = patches
        self.motion_type = motion_type
        self.nsteps = nsteps

        # -- create frame sampler --
        B,R,T,H,C,H,W = patches.shape
        self.frame_sampler = FrameIndexSampler(T,H)
        self.block_sampler = BlockIndexSampler(T,H)

        # -- samples of nh indices for each frame --
        self.samples = {t:[] for t in range(patches.shape[2])}
        
    
    def get_ref_h(self):
        H = int(np.sqrt(self.patches.shape[3]))
        return H**2//2 + H//2*(NH%2==0)

    def step(self,search_frames,fixed_frames):
        # -- setup --
        # patches = self.patches
        # B,R,T,H,C,H,W = patches.shape
        # ref = patches[:,:,T//2,self.ref_h,:,:,:]
        # search = patches[:,:,search_frames,:,:,:,:]
        # fixed = patches[:,:,fixed_frames,:,:,:,:]

        # -- select sampling method --
        if self.motion_type == "global":
            samples = self.global_motion_step(search_frames,fixed_frames)
        elif self.motion_type == "local":
            raise NotImplemented("No local motion working yet!")
        else:
            raise ValueError(f"Uknown motion type [{self.motion_type}]")

        self.update(samples,search_frames,fixed_frames)
        return samples

    def global_motion_step(self,search_frames,fixed_frames):

        # -- move vars into scope --
        patches = self.patches
        B,R,T,H,C,H,W = patches.shape
        ref = patches[:,:,T//2,self.ref_h,:,:,:]

        # -- sample the grid --
        block_sampler = self.block_sampler.set_frames(search_frames,fixed_frames)

        # -- compute along grid --
        scores = torch.zeros(search_frames.shape[0])
        while not block_sampler.terminated():

            # -- sample a neighborhood grid --
            nh_grid = block_sampler.sample(scores)

            # -- index the patch for the neighborhood --
            nh_patches = patches[:,:,nh_grid,:,:,:] 

            # -- compute the scores per search frame --
            scores = self.compute_frame_scores(ref,nh_patches)

            block_sampler.update(scores,nh_grid)

        samples = block_sampler.get_results()
        return samples

    def compute_frame_scores(self,ref,nh_patches):

        # -- include the reference patch --
        aug_patches = torch.cat([ref,nh_patches],dim=2)
        for (nset0,nset1) in n_grids[:100]:

            # -- original --
            denoised0 = torch.mean(grid_patches[:,:,nset0],dim=2)
            denoised1 = torch.mean(grid_patches[:,:,nset1],dim=2)


    def update(self,samples,search,fixed):
        pass

    def get_best_samples(self,K=3):

        # -- do the following for each conditional dist.'s set of samples --
        aggregate = self.aggregate
        aggregate = pd.DataFrame(aggregate)
        block_freqs = aggregate.value_counts()
        # -- sure, K could be sampled --
        topK = np.array(block_freqs.index[:K])
        block_indices = block_str_to_list(topK)
        return block_indices

