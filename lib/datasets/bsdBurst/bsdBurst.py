"""
BSD Burst Dataset from
"Fast Burst Image Denoising"

"""

# -- python imports --
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
from pyutils.timer import Timer
from datasets.transforms import get_dynamic_transform,get_noise_transform

# -- local imports --
from .readers import read_bsdBurst_paths,read_bsdBurst_burst

# -------------------
#   Dataset Object
# -------------------

class BSDBurst():

    def __init__(self,root,noise_info,frame_size,nframes):

        # -- init vars --
        self.root = root
        self.noise_info = noise_info
        self.frame_size = frame_size
        self.nframes = nframes

        # -- input checking for num of frames --
        if nframes is None: self.nframes = 10
        if self.nframes > 10:
            print("[bsdBurst] WARNING: too many frames set in config. MAX = 10")
            self.nframes = 10

        # -- create transforms --
        self.noise_xform = get_noise_transform(noise_info,use_to_tensor=False)

        # -- retrieve paths --
        self.dataset,self.bnames = self._read_dataset_paths(self.root)

    def _read_dataset_paths(self,path):
        return read_bsdBurst_paths(path)

    def _set_random_state(self,rng_state):
        torch.set_rng_state(rng_state['th'])
        np.random.set_state(rng_state['np'])

    def _get_random_state(self):
        th_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()
        rng_state = edict({'th':th_rng_state,'np':np_rng_state})
        return rng_state

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # -- get random state --
        rng_state = self._get_random_state()

        # -- read sample --
        name = self.bnames[index]
        paths = self.dataset[name]
        data = read_bsdBurst_burst(paths,self.frame_size,self.nframes)

        # -- extract data --
        burst = torch.FloatTensor(data['burst'])
        nnf_vals = torch.FloatTensor(data['nnf_vals'])
        nnf_locs = torch.FloatTensor(data['nnf_locs'])

        # -- pick only top 1 --
        nnf_vals = nnf_vals[...,0]
        nnf_locs = nnf_locs[...,0,:]

        # -- rename clean frames --
        nframes = len(burst)
        clean = burst

        # -- create noisy sample --
        noisy = [self.noise_xform(img) for img in clean]
        noisy = torch.stack(noisy,dim=0)

        # -- create static iid noisy samples --
        sclean = repeat(clean[nframes//2],'c h w -> t c h w',t=nframes)
        snoisy = [self.noise_xform(burst[nframes//2]) for l in range(nframes)]
        snoisy = torch.stack(snoisy,dim=0)
        index = torch.IntTensor([index])

        # -- create sample --
        sample = {'dyn_clean':clean,'dyn_noisy':noisy,
                  'static_clean':sclean,'static_noisy':snoisy,
                  'nnf_vals':nnf_vals,'nnf':nnf_locs,
                  'nnf_locs':nnf_locs,'image_index':index,
                  'flow':nnf_locs,'occ':None,'rng_state':rng_state}
        return sample




