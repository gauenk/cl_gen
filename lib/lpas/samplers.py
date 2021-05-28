
# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
from .utils import get_ref_block_index,get_block_arangements_freeze

class FrameIndexSampler():
    """
    Select the groups of frames to use for optimization
    """

    def __init__(self,nframes,nblocks,ref_frame,mtype,motion=None):
        self.nframes = nframes
        self.nblocks = nblocks
        self.ref_frame = ref_frame
        self.ref_block = nblocks**2//2
        self.mtype = mtype
        self.motion = motion
        self.fixed_frames = edict({'idx':[self.ref_frame],'vals':[self.ref_block]})

    def init_sample(self):
        self.current = 0
        return self.sample()

    def sample(self):
        # self.current += 1
        # if self.current == self.ref_frame: self.current += 1
        # self.current = self.current % self.nframes
        # sample = np.array([self.current])
        self.reset_fixed_frames()
        # self.append_fixed_frames(self.current,self.ref_block)
        return self.fixed_frames

    def fixed(self,sampled):
        fixed = list(np.arange(self.nframes))
        fixed = [fixed.remove(x) for x in list(sampled)]
        return np.array(fixed)

    def reset_fixed_frames(self):
        self.fixed_frames = edict({'idx':[self.ref_frame],'vals':[self.ref_block]})

    def append_fixed_frames(self,idx,val):
        self.fixed_frames.idx.append(idx)
        self.fixed_frames.vals.append(val)

class MotionSampler():

    def __init__(self,nframes,nblocks,ref_frame,isize,mtype='global_const',vectors=None):
        self.nframes = nframes
        self.nblocks = nblocks
        self.ref_frame = ref_frame
        self.isize = isize
        self.mtype = mtype
        self.samples = edict({'scores':[],'vectors':[]})

    def init_sample(self):
        if self.mtype == 'global_const':
            return torch.zeros(1,1,2)
        elif self.mtype == 'global_jitter':
            return torch.zeros(1,self.nframes,2)
        elif self.mtype == 'local_const':
            return torch.zeros(self.isize,1,2)
        elif self.mtype == 'local_jitter':
            return torch.zeros(self.isize,self.nframes,2)
        else:
            raise ValueError(f"Uknown motion type [{self.mytpe}]")

    def sample(self,scores,vectors):
        # -- add to data --
        self.append_samples(scores,vectors)

        # -- propose new motion vector --
        return vectors

    def append_samples(self,scores,vectors):
        self.samples.scores.append(scores)
        self.samples.vectors.append(vectors)

class BlockIndexSampler():

    def __init__(self,nframes,nblocks,ref_frame,isize,mtype,motion=None):
        self.nframes = nframes
        self.nblocks = nblocks
        self.ref_frame = ref_frame
        self.isize = isize
        self.mtype = mtype
        self.motion = motion
        self._terminated = False
        self.samples = edict({'scores':[],'block_grids':[]})
        # self.full_grid = get_block_arangements_freeze(nblocks,self.fixed_frames)

    def create_full_grid(self):
        pass

    def init_sample(self):
        nblocks,nframes = self.nblocks,self.nframes
        block_grids = torch.LongTensor([[nblocks**2//2 for t in range(nframes)]])        
        return block_grids

    def terminated(self):
        return self._terminated

    def __getitem__(self,index):
        pass

    def sample(self,current_frames,fixed_frames,motion):
        gbaf = get_block_arangements_freeze
        if isinstance(fixed_frames,list):
            B = len(fixed_frames)
            grids = []
            for b in range(B):
                grid = gbaf(self.nframes,self.nblocks,fixed_frames[b])
                grids.append(grid)
            grids = torch.stack(grids,dim=1)
            return grids
        else:
            grids = gbaf(self.nframes,self.nblocks,fixed_frames)
            return grids
        # if gtype == "mesh":
        #     grid = get_block_arangements_freeze(self.nframes,self.nblocks,fixed_frames)
        # elif gtype == "split":
        #     grid = get_block_arangements_split(current_frames,self.nframes,
        #                                        self.nblocks,fixed_frames)
        # else:
        #     raise ValueError(f"Unknown grid type [{gtype}]")
        return grid

    def reset(self):
        self.samples.scores = []
        self.samples.block_grids = []

    def update(self,scores,block_grid):

        # -- expand block_grid if necessary --
        ndims = len(block_grid.shape)
        block_batch_dim = (ndims == 3)
        if not block_batch_dim:
            if ndims == 1: block_grid = block_grid[None,None,:]
            elif ndims == 2: block_grid = block_grid[:,None,:]

        # -- we can also use torch.cat, todo later --
        E = scores.shape[1]
        for e in range(E):
            self.samples.scores.append(scores[:,e])
            self.samples.block_grids.append(block_grid[:,e])

    def get_results(self,K=3):
        scores = torch.stack(self.samples.scores,dim=0)
        block_grids = torch.stack(self.samples.block_grids,dim=0)
        # print("[block_sampler.update]:",scores.shape,block_grids.shape)
        N,B,Tp1 = scores.shape
        same_grids = (B == block_grids.shape[1])
        if N < K: K = N
        scores_topk,block_grids_topk = [],[]
        for b in range(B):
            order_b = torch.topk(scores[:,b,0],K,largest=False).indices
            scores_topk.append(scores[:,b,0][order_b])
            if not same_grids: # block grid eq for all batch
                block_grids_topk.append(block_grids[order_b,:]) 
            else:
                block_grids_topk.append(block_grids[order_b,b])                 
        scores_topk = torch.stack(scores_topk,dim=0)
        block_grids_topk = torch.stack(block_grids_topk,dim=0)

        return scores_topk,block_grids_topk
        
    def get_samples(self):
        pass

