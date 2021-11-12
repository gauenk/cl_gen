"""
CBSD68 dataset

"""

# python imports
import pdb,glob
from PIL import Image
from functools import partial
from easydict import EasyDict as edict
import numpy.random as npr
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

# pytorch imports
import torch,torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as th_transforms
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import DatasetFolder

# -- project imports --
from datasets.common import get_loader,return_optional
from datasets.transforms import get_dynamic_transform,get_noise_transform
from datasets.reproduce import RandomOnce,get_random_state,enumerate_indices

class Template():

    def __init__(self,root,split,isize,nsamples,noise_info,dynamic_info):

        # -- set init params --
        self.root = root
        self.split = split
        self.noise_info = noise_info
        self.dynamic_info = dynamic_info
        self.nsamples = nsamples
        self.isize = isize

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)
        self.dynamic_trans = get_dynamic_transform(dynamic_info,None,load_res)

        # -- load paths --
        self.paths = []

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths),nsamples)
        self.nsamples = len(self.indices)

        # -- single random dynamics --
        self.dyn_once = return_optional(dynamic_info,"sim_once",False)
        self.fixRandDynamics = RandomOnce(self.dyn_once,nsamples)

        # -- single random noise --
        self.noise_once = return_optional(noise_info,"sim_once",False)
        self.fixRandNoise_1 = RandomOnce(self.noise_once,nsamples)
        self.fixRandNoise_2 = RandomOnce(self.noise_once,nsamples)

    def __len__(self):
        return self.nsamples                

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # -- get random state --
        rng_state = self._get_random_state()
        
        # -- image --
        image_index = self.indices[index]
        img = Image.open(self.images[image_index]).convert("RGB")

        # -- get dynamics ---
        with self.fixRandDynamics.set_state(index):
            dyn_clean,res_set,_,seq_flow,ref_flow,tl = self.dynamic_trans(img)

        # -- get noise --
        with self.fixRandNoise_1.set_state(index):
            dyn_noisy = self.noise_trans(dyn_clean)#+0.5
        nframes,c,h,w = dyn_noisy.shape

        # -- get second, different noise --
        static_clean = repeat(dyn_clean[nframes//2],'c h w -> t c h w',t=nframes)
        with self.fixRandNoise_2.set_state(index):
            static_noisy = self.noise_trans(static_clean)#+0.5

        # -- manage flow and output --
        ref_flow = repeat(ref_flow ,'t two -> t h w two',h=h,w=w)
        index_th = torch.IntTensor([image_index])

        return {'dyn_noisy':dyn_noisy,'dyn_clean':dyn_clean,
                'static_noisy':static_noisy,'static_clean':static_clean,
                'nnf':ref_flow,'seq_flow':seq_flow, 'ref_flow':ref_flow,
                'flow':ref_flow,'index':index_th,'rng_state':rng_state}


#
# Loading the datasets in a project
#

def get_cbsd68_dataset(cfg):

    #
    # -- extract --
    #

    # -- misc --
    root = cfg.dataset.root
    noise_info = cfg.noise_params[cfg.noise_type]
    dynamic_info = cfg.dynamic
    isize = return_optional(cfg,"frame_size",None)

    # -- samples --
    nsamples = return_optional(cfg,"nsamples",0)
    tr_nsamples = return_optional(cfg,"tr_nsamples",nsamples)
    val_nsamples = return_optional(cfg,"val_nsamples",nsamples)
    te_nsamples = return_optional(cfg,"te_nsamples",nsamples)

    # -- setup paths --
    root = Path(root)/Path("./template/images/")

    # -- create objcs --
    data = edict()
    data.tr = Template(root,"train",isize,tr_nsamples,noise_params,dynamic_info)
    data.val = Template(root,"val",isize,val_samples,noise_params,dynamic_info)
    data.test = Template(root,"test",isize,te_samples,noise_params,dynamic_info)

    # -- create loader --
    loader = get_loader(cfg,data,cfg.batch_size)

    return data,loader


