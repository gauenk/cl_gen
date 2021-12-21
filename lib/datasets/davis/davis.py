"""
DAVIS dataset

"""

# -- python imports --
import pdb
import numpy as np
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
from datasets.common import get_loader,return_optional
from datasets.transforms import get_noise_transform
from datasets.reproduce import RandomOnce,get_random_state,enumerate_indices

# -- local imports --
from .paths import IMAGE_PATH,FLOW_PATH,IMAGE_SETS
from .reader import read_files,read_subburst_files,read_burst,read_pix,pix2flow

class DAVIS():

    def __init__(self,iroot,froot,sroot,split,isize,ps,nsamples,nframes,noise_info):

        # -- set init params --
        self.iroot = iroot
        self.froot = froot
        self.sroot = sroot
        self.split = split
        self.noise_info = noise_info
        self.ps = ps
        self.nsamples = nsamples
        self.isize = isize

        # -- create transforms --
        self.noise_trans = get_noise_transform(noise_info,noise_only=True)

        # -- load paths --
        self.paths,self.nframes,all_eq = read_subburst_files(iroot,froot,sroot,split,
                                                             isize,ps,nframes)
        # self.paths,self.nframes,all_eq = read_files(iroot,froot,sroot,split,
        #                                             isize,ps,nframes)
        # msg = "\n\n\n\nWarning: Not all bursts are same length!!!\n\n\n\n"
        # if not(all_eq): print(msg)
        self.groups = sorted(list(self.paths['images'].keys()))

        # -- limit num of samples --
        self.indices = enumerate_indices(len(self.paths['images']),nsamples)
        self.nsamples = len(self.indices)
        nsamples = self.nsamples
        print("nsamples: ",nsamples)

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
        rng_state = get_random_state()

        # -- indices --
        image_index = self.indices[index]
        group = self.groups[image_index]
        tframes = len(self.paths['images'][group])
        nframes = tframes if self.nframes is None else self.nframes

        # -- select correct image paths --
        ref = tframes//2
        start = ref - nframes//2
        frame_ids = np.arange(start,start+nframes)

        # -- load burst --
        img_fn = self.paths['images'][group]
        icrop = self.paths['crops'][group]
        dyn_clean = read_burst(img_fn,self.isize,icrop)

        # -- load pix & flow --
        ref_pix = read_pix(self.paths['flows'][group])
        ref_flow = pix2flow(ref_pix)

        # -- format pix --
        ref_pix = rearrange(ref_pix,'two t k h w -> k t h w two')
        ref_pix = torch.LongTensor(ref_pix)#.copy())

        # -- format flow --
        ref_flow = rearrange(ref_flow,'two t k h w -> k t h w two')
        ref_flow = torch.FloatTensor(ref_flow.copy())#.copy())

        # -- get noise --
        with self.fixRandNoise_1.set_state(index):
            dyn_noisy = self.noise_trans(dyn_clean)#+0.5
        nframes,c,h,w = dyn_noisy.shape

        # -- get second, different noise --
        static_clean = repeat(dyn_clean[nframes//2],'c h w -> t c h w',t=nframes)
        with self.fixRandNoise_2.set_state(index):
            static_noisy = self.noise_trans(static_clean)#+0.5

        # -- manage flow and output --
        index_th = torch.IntTensor([image_index])

        return {'dyn_noisy':dyn_noisy,'dyn_clean':dyn_clean,
                'static_noisy':static_noisy,'static_clean':static_clean,
                'nnf':ref_flow,'seq_flow':None, 'ref_flow':ref_flow,
                'flow':ref_flow,'index':index_th,'rng_state':rng_state,
                'ref_pix':ref_pix}


#
# Loading the datasets in a project
#

def get_davis_dataset(cfg):

    #
    # -- extract --
    #

    # -- noise and dyanmics --
    noise_info = cfg.noise_params
    isize = return_optional(cfg,"frame_size",None)
    nframes = return_optional(cfg,"nframes",None)
    ps = return_optional(cfg,"patchsize",1)

    # -- samples --
    nsamples = return_optional(cfg,"nsamples",0)
    tr_nsamples = return_optional(cfg,"tr_nsamples",nsamples)
    val_nsamples = return_optional(cfg,"val_nsamples",nsamples)
    te_nsamples = return_optional(cfg,"te_nsamples",nsamples)

    # -- setup paths --
    iroot = IMAGE_PATH
    froot = FLOW_PATH
    sroot = IMAGE_SETS

    # -- create objcs --
    data = edict()
    data.tr = DAVIS(iroot,froot,sroot,"train",isize,ps,tr_nsamples,nframes,noise_info)
    data.val = DAVIS(iroot,froot,sroot,"val",isize,ps,val_nsamples,nframes,noise_info)
    data.te = DAVIS(iroot,froot,sroot,"test",isize,ps,te_nsamples,nframes,noise_info)

    # -- create loader --
    loader = get_loader(cfg,data,cfg.batch_size)

    return data,loader


