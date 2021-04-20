
import numpy as np
from easydict import EasyDict as edict

class FrameIndexSampler():

    def __init__(self,T,H,motion=None):
        self.T = T
        self.H = H
        self.motion = motion

    def init_sample(self):
        
        pass

    def sample(self):
        pass

    def fixed(self,sampled):
        pass

class MotionSampler():

    def __init__(self,T,H,mtype='global',vectors=None):
        self.T = T
        self.H = H
        self.mtype = mtype
        self.vectors = vectors
        self.samples = edict({'scores':[],'vectors':[]})

    def init_sample(self):
        if self.mtype == 'global':
            return torch.ones(1,2)
        elif self.mtype == 'local':
            return torch.ones((self.T,2))

    def sample(self,scores):
        scores = self.samples.scores[-1]
        current = self.vectors
        vectors = self.vectors
        self.samples.scores.append(scores)
        self.samples.vectors.append(vectors)
        return vectors

class BlockIndexSampler():

    def __init__(self,T,H,motion=None):
        self.T = T
        self.H = H
        self.motion = motion
        self._terminated = False
        self.search_frames = None
        self.fixed_frames = None
        self.full_grid = self.create_full_grid()

    def set_frames(search_frames,fixed_frames):
        self.search_frames = search_frames
        self.fixed_frames = fixed_frames

    def create_full_grid(self):
        pass

    def terminated(self):
        return self._terminated

    def __getitem__(self,index):
        pass
        
    def sampler(self,scores):
        pass

    def update(self,scores,nh_grid):
        pass

    def get_samples(self):
        pass


