class FrameIndexSampler():

    def __init__(self,T,H):
        pass

    def init_sample(self):
        pass

    def sample(self):
        pass

    def fixed(self,sampled):
        pass

class BlockIndexSampler():

    def __init__(self,T,H):
        self.T = T
        self.H = H
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


