
# -- python imports --
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange

# -- pytorch imports --
import torch

# -- project imports --
from layers.unet import UNet_n2n,UNet_small

# -- [local] project imports --
from .cog import COG
from .scores import get_score_function
from .samplers import FrameIndexSampler,BlockIndexSampler,MotionSampler

class AlignmentOptimizer():

    def __init__(self,T,H,isize,motion_type,nsteps,nparticles,motion,score_params):
        # -- init vars --
        self.nframes = T
        self.nblocks = H
        self.isize = isize
        self.motion_type = motion_type
        self.nsteps = nsteps
        self.nparticles = nparticles
        self.motion = motion
        self.score_params = score_params

        # -- create samplers --
        self.frame_sampler = FrameIndexSampler(T,H,motion_type,motion)
        self.block_sampler = BlockIndexSampler(T,H,isize,motion_type,motion)
        self.motion_sampler = MotionSampler(T,H,isize,motion_type,motion)

        # -- samples of nh indices for each frame --
        self.samples = edict({'scores':[],'blocks':[]})
        
    
    def get_ref_h(self):
        Hsqrt = int(np.sqrt(self.nblocks))
        return Hsqrt**2//2 + Hsqrt//2*(Hsqrt%2==0)

    def sample(self,patches,block_grids):
        self.parallel_limit = -1 #10000
        B,R,T,H,C,H,W = patches.shape        
        if B*R*(H**2) < self.parallel_limit:
            return self.compute_vectorized_block_grid(patches,block_grids)
        else:
            return self.compute_serial_block_grid(patches,block_grids)

    def compute_vectorized_block_grid(self,patches,block_grids):
        raise NotImplemented("")

    def compute_serial_block_grid(self,patches,block_grids):
        # -- move vars into scope --
        B,R,T,H,C,H,W = patches.shape
        tgrid = torch.arange(T)

        # -- compute along grid --
        scores = torch.zeros(self.nframes)
        self.block_sampler.reset()
        for blocks in block_grids:

            # -- index the patch for the neighborhood --
            block = patches[:,:,tgrid,blocks,:,:,:] 
            block = block.unsqueeze(2)

            # -- compute the scores per search frame --
            scores = self.compute_frame_scores(block)
            # B,E,Tp1 = scores.shape

            # -- update block sampler --
            blocks = blocks.cuda(non_blocking=True)[None,:]
            # print(scores[:,:,0],blocks)
            self.block_sampler.update(scores,blocks)
            """
            if we include block sampler in this block
            we can adapt the search grid during the local search.
            """
        scores,blocks = self.block_sampler.get_results()
        self.append_samples(scores,blocks)
        return scores,blocks

    def compute_frame_scores(self,block):
        stype = self.score_params.stype
        if stype == "raw":
            return self.compute_raw_frame_scores(block)
        elif stype == "cog":
            return self.compute_cog_frame_scores(block)
        else:
            raise ValueError(f"Uknown score type [{stype}]")

    def compute_cog_frame_scores(self,ref,bl_patches):
        cog = COG(UNet_small,T,noisy.device,nn_params=None,train_steps=train_steps)
        cog.train_models(noisy_prop)
        recs = cog.test_models(noisy_prop)
        score = cog.operator_consistency(recs,noisy_prop)
        scores_t = repeat(score,'r b e c h w -> r b e t c h w',t=T)
        scores = self.aggregate_scores(score,scores_t)
        return scores

    def compute_raw_frame_scores(self,block):
        # R,B,E,T,C,H,W = block.shape
        score_fxn = get_score_function(self.score_params.name)
        score,scores_t = score_fxn(None,block)
        score = torch.mean(score,dim=0)
        scores_t = torch.mean(scores_t,dim=0)
        scores = self.aggregate_scores(score,scores_t)
        return scores

    def aggregate_scores(self,score,scores_t):
        score = score.unsqueeze(2)
        scores = torch.cat([score,scores_t],dim=2)
        return scores

    def append_samples(self,scores,blocks):
        K = scores.shape[1]
        for k in range(K):
            self.samples.scores.append(scores[:,k])
            self.samples.blocks.append(blocks[:,k])

    def get_best_samples(self,K=3):

        # -- do the following for each conditional dist.'s set of samples --
        scores = torch.stack(self.samples.scores,dim=0)
        blocks = torch.stack(self.samples.blocks,dim=0)

        # -- reshape --
        scores = rearrange(scores,'n b -> b n')
        blocks = rearrange(blocks,'n b g -> b n g')

        # -- pick top K --
        B = scores.shape[0]
        scores_topK,blocks_topK = [],[]
        for b in range(B):
            topK = torch.topk(scores[b],K,largest=False)
            scores_topK.append(topK.values)
            blocks_topK.append(blocks[b,topK.indices])
        scores_topK = torch.stack(scores_topK,dim=0)
        blocks_topK = torch.stack(blocks_topK,dim=0)
        return scores_topK,blocks_topK

