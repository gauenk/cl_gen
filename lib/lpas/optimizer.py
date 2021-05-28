
# -- python imports --
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch imports --
import torch

# -- project imports --
from layers.unet import UNet_n2n,UNet_small
from patch_search import get_score_function
from patch_search.cog import COG

# -- [local] project imports --
from .samplers import FrameIndexSampler,BlockIndexSampler,MotionSampler

class AlignmentOptimizer():

    def __init__(self,T,H,refT,isize,motion_type,nsteps,nparticles,motion,score_params):
        # -- init vars --
        self.nframes = T
        self.nblocks = H
        self.ref_frame = refT
        self.isize = isize
        self.motion_type = motion_type
        self.nsteps = nsteps
        self.nparticles = nparticles
        self.motion = motion
        self.score_params = score_params
        self.verbose = False

        # -- create samplers --
        self.frame_sampler = FrameIndexSampler(T,H,refT,motion_type,motion)
        self.block_sampler = BlockIndexSampler(T,H,refT,isize,motion_type,motion)
        self.motion_sampler = MotionSampler(T,H,refT,isize,motion_type,motion)

        # -- samples of nh indices for each frame --
        self.init_samples()
        self.igrid = edict()
        
    
    def init_samples(self):
        self.samples = edict({'scores':[],'blocks':[]})

    def get_ref_h(self):
        # Hsqrt = int(np.sqrt(self.nblocks))
        # return Hsqrt**2//2 + Hsqrt//2*(Hsqrt%2==0)
        H = self.nblocks
        return H**2//2 + H//2*(H%2==0)

    def sample(self,patches,block_grids,K):
        self.parallel_limit = -1 #10000
        B,R,T,H,C,H,W = patches.shape        
        if B*R*(H**2) < self.parallel_limit:
            return self.compute_vectorized_block_grid(patches,block_grids,K)
        else:
            return self.compute_serial_block_grid(patches,block_grids,K)

    def compute_vectorized_block_grid(self,patches,block_grids,K):
        raise NotImplemented("")

    def _index_block_from_patch(self,patches,blocks,indexing):
        i = indexing
        ndims = len(blocks.shape)
        if ndims >= 2: # -- image batch dim present --
            bbs = blocks.shape[0]
            #torch.index_select(patches,
            block = patches[i.bmesh[:,:bbs],:,i.tmesh[:,:bbs],blocks,:,:,:] 
            block = rearrange(block,'b e t r c ph pw -> b r e t c ph pw')
        else: # -- no image batch dim --
            block = patches[:,:,i.tmesh[0],blocks,:,:,:] 
            block = block.unsqueeze(2)
        return block

    def block_grid_loader(self,block_grids,batchsize):
        nbatches = len(block_grids) // batchsize
        nbatches += len(block_grids) % batchsize > 0
        for i in range(nbatches):
            start = i * batchsize
            end = start + batchsize
            batch = rearrange(block_grids[start:end],'e b t -> b e t') # E, B, T 
            yield batch

    def compute_serial_block_grid(self,patches,block_grids,K):
        # -- move vars into scope --
        B,R,T,H,C,Ph,Pw = patches.shape
        BB = 100

        # -- create indexing grids
        tgrid = None
        bmesh = repeat(torch.arange(B),'b -> b e t',t=T,e=BB)
        tmesh = repeat(torch.arange(T),'t -> b e t',b=B,e=BB)
        indexing = edict({'tgrid':tgrid,'bmesh':bmesh,'tmesh':tmesh})

        # -- compute along grid --
        scores = torch.zeros(self.nframes)
        self.block_sampler.reset()
        if len(block_grids.shape) == 2: block_grids = block_grids[:,None,:]
        # print("P",patches.shape)
        for blocks in self.block_grid_loader(block_grids,BB):
            """
            todo: modify "blocks" shape here once and for all. 
                  no more switching elsewhere
            """
            if len(blocks.shape) <= 2: blocks = blocks[None,:]
            # blocks = repeat(blocks,'1 b t -> b e t',e=3)

            # -- index the patch for the neighborhood --
            block = self._index_block_from_patch(patches,blocks,indexing)
            # B,R,E,T,C,H,W = blocks.shape, E = batches of block grids

            # -- compute the scores per search frame --
            if self.verbose: print(blocks)
            scores = self.compute_frame_scores(block)
            # B,E,Tp1 = scores.shape

            # print("bs.shape",blocks.shape)
            # print("s.shape",scores.shape)
            # print("b.shape",block.shape)

            # -- update block sampler --
            #blocks = blocks.cuda(non_blocking=True)
            self.block_sampler.update(scores,blocks)
            """
            if we include block sampler in this block
            we can adapt the search grid during the local search.
            """
        scores,blocks = self.block_sampler.get_results(K=K)
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
        # B,R,E,T,C,H,W = block.shape
        block = rearrange(block,'b r e t c h w -> r b e t c h w')
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
        B,N = scores.shape
        if N < K: K = N
        scores_topK,blocks_topK = [],[]
        for b in range(B):
            topK = torch.topk(scores[b],K,largest=False)
            scores_topK.append(topK.values)
            blocks_topK.append(blocks[b,topK.indices])
        scores_topK = torch.stack(scores_topK,dim=0)
        blocks_topK = torch.stack(blocks_topK,dim=0)
        return scores_topK,blocks_topK

# class BlockGridLoader():
#     def __init__(self,block_grids,batchsize):
#         self.block_grids
#         self.batchsize

#         # -- compute num of batches --
#         self.nbatches = len(self.block_grids) // self.batchsize
#         self.nbatches += self.batchsize % len(self.block_grids) > 0
        
#     def __len__(self):
#         return self.nbatches
#     def __next__(self):
#         return self.next()

#     def next(self):
#         for i in range(self.nbatches):
#             start = i*self.batchsize
#             yield
