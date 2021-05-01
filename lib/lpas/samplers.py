
# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
from .utils import get_ref_block_index,get_block_arangements_freeze

class FrameIndexSampler():

    def __init__(self,nframes,nblocks,mtype,motion=None):
        self.nframes = nframes
        self.nblocks = nblocks
        self.ref_frame = nframes//2
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

    def __init__(self,nframes,nblocks,isize,mtype='global_const',vectors=None):
        self.nframes = nframes
        self.nblocks = nblocks
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

    def __init__(self,nframes,nblocks,isize,mtype,motion=None):
        self.nframes = nframes
        self.nblocks = nblocks
        self.isize = isize
        self.mtype = mtype
        self.motion = motion
        self._terminated = False
        self.samples = edict({'scores':[],'block_grids':[]})
        # self.full_grid = get_block_arangements_freeze(nblocks,self.fixed_frames)

    def create_full_grid(self):
        pass

    def terminated(self):
        return self._terminated

    def __getitem__(self,index):
        pass

    def sample(self,fixed_frames,motion):
        grid = get_block_arangements_freeze(self.nframes,self.nblocks,fixed_frames)
        return grid

    def reset(self):
        self.samples.scores = []
        self.samples.block_grids = []

    def update(self,scores,block_grid):
        E = scores.shape[1]
        for e in range(E):
            self.samples.scores.append(scores[:,e])
            self.samples.block_grids.append(block_grid[e])

    def get_results(self,K=3):
        scores = torch.stack(self.samples.scores,dim=0)
        block_grids = torch.stack(self.samples.block_grids,dim=0)
        N,B,Tp1 = scores.shape
        if N < K: K = N
        scores_topk,block_grids_topk = [],[]
        for b in range(B):
            order_b = torch.topk(scores[:,b,0],K,largest=False).indices
            scores_topk.append(scores[:,b,0][order_b])
            block_grids_topk.append(block_grids[order_b,:]) # block grid same for all batch
        scores_topk = torch.stack(scores_topk,dim=0)
        block_grids_topk = torch.stack(block_grids_topk,dim=0)

        return scores_topk,block_grids_topk
        
    def get_samples(self):
        pass

