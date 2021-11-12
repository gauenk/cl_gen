"""
The KITTI dataset with flow?

"""


# -- python imports --
import os,cv2,tqdm
import numpy as np
import numpy.random as npr
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils import print_tensor_stats
from datasets.common import get_loader,return_optional
from datasets.kitti.burst_with_flow_reader import read_dataset_paths,read_dataset_testing,read_dataset_sample
from datasets.transforms import get_noise_transform
from datasets.kitti.paths import get_kitti_path

class BurstWithFlowKITTI():

    def __init__(self,root,split,edition,nframes,
                 noise_info,crop,resizes,
                 nnf_K,nnf_ps,nnf_exists):
        self.root = root
        paths = get_kitti_path(root)
        self.edition = edition
        self.paths = paths
        self.split = split
        self.istest = split == "test"
        self.nframes = nframes
        self.noise_info = noise_info
        self.crop = crop
        self.resizes = resizes
        self.nnf_K = nnf_K
        self.nnf_ps = nnf_ps
        self.nnf_exists = nnf_exists
        parts = self._get_split_parts_name(split)
        self.dataset = self._read_dataset_paths(paths,edition,parts,
                                                nframes,self.resizes.path,
                                                nnf_K,nnf_ps,nnf_exists)
        self.noise_xform = get_noise_transform(noise_info,use_to_tensor=False)

    def _read_dataset_paths(self,paths,edition,parts,nframes,read_resize,
                            nnf_K,nnf_ps,nnf_exists):
        if parts in ["train","val","mixed"]:
            return read_dataset_paths(paths,edition,parts,nframes,
                                      resize=read_resize,
                                      nnf_K = nnf_K, nnf_ps = nnf_ps,
                                      nnf_exists = nnf_exists)
        elif parts in ["test"]:
            return read_dataset_testing(paths,edition,nframes,
                                        resize=read_resize,
                                        nnf_K = nnf_K, nnf_ps = nnf_ps,
                                        nnf_exists = nnf_exists)
        else: raise ValueError(f"[KITTI: read_dataset] Uknown part [{parts}]")
            
    def _read_dataset_sample(self,burst_id,ref_t,edition):
        data = read_dataset_sample(burst_id,ref_t,self.nframes,edition,
                                   self.istest,path=self.paths,
                                   crop = self.crop,
                                   resizes = self.resizes,
                                   nnf_K = self.nnf_K,
                                   nnf_ps = self.nnf_ps)
        return data

    def _get_split_parts_name(self,split):
        if split == "trainval": parts = "mixed"
        elif split == "train": parts = "train"
        elif split == "val": parts = "val"
        elif split == "test": parts = "test"
        else: raise ValueError(f"[KITTI: get_split_parts_name] Uknown split [{split}]")
        return parts

        
    def _set_random_state(self,rng_state):
        torch.set_rng_state(rng_state['th'])
        np.random.set_state(rng_state['np'])

    def _get_random_state(self):
        th_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()
        rng_state = edict({'th':th_rng_state,'np':np_rng_state})
        return rng_state
        
    def __len__(self):
        return len(self.dataset['burst_id'])

    def __getitem__(self,index):
        
        # -- get random state --
        rng_state = self._get_random_state()

        # -- read sample --
        burst_id = self.dataset['burst_id'][index]
        ref_t = self.dataset['ref_t'][index]
        edition = self.dataset['edition'][index]
        data = self._read_dataset_sample(burst_id,ref_t,edition)

        # -- extract data --
        burst = torch.FloatTensor(data['burst'])
        ref_frame = torch.FloatTensor(data['ref_frame'])
        flows = torch.FloatTensor(data['flows'])
        occs = torch.FloatTensor(data['occs'])
        nnf_vals = torch.FloatTensor(data['nnf_vals'])
        nnf_locs = torch.FloatTensor(data['nnf_locs'])
        burst = rearrange(burst,'t h w c -> t c h w')

        # -- rename clean frames --
        nframes = len(burst)
        clean = burst/255.

        # -- create noisy sample --
        noisy = [self.noise_xform(img)+0.5 for img in clean]
        noisy = torch.stack(noisy,dim=0)

        # -- create static iid noisy samples --
        sclean = repeat(clean[nframes//2],'c h w -> t c h w',t=nframes)
        snoisy = [self.noise_xform(burst[nframes//2])+0.5 for l in range(nframes)]
        snoisy = torch.stack(snoisy,dim=0)

        # -- make index a tensor --
        index_th = torch.IntTensor([index])
        
        # -- create sample --
        sample = {'dyn_clean':clean,'dyn_noisy':noisy,
                  'static_clean':sclean,'static_noisy':snoisy,
                  'nnf_vals':nnf_vals,'nnf':nnf_locs,
                  'nnf_locs':nnf_locs,'image_index':index_th,
                  'flows':flows,'occs':occs,'rng_state':rng_state,}
        return sample

def write_burst_with_flow_kitti_nnf(cfg):
    cfg.dataset.nnf_exists = False
    data,loader = get_burst_with_flow_kitti_dataset(cfg,"dynamic")
    # -- iterating through dataset will write the nnf --
    for split in data.keys():
        for i in tqdm.tqdm(range(len(data[split]))):
            data[split][i] # this will write the missing nnfs 

def get_burst_with_flow_kitti_dataset(cfg,mode):
    root = cfg.dataset.root
    root = Path(root)/Path("kitti")
    data = edict()
    batch_size = cfg.batch_size
    rtype = 'dict' if cfg.dataset.dict_loader else 'list'
    nnf_K = 3
    nnf_ps = 3
    nnf_exists = return_optional(cfg.dataset,'nnf_exists',True)

    crop = cfg.frame_size
    if isinstance(crop,int):
        print("WARNING: the [cfg.frame_size] parameter is just an int.")
    path_resize = (1224, 370)
    load_resize = crop#(256, 128)
    resizes = edict({'path':path_resize,'load':load_resize})

    if mode == "dynamic":
        edition = "2015"
        nframes = cfg.nframes
        noise_info = cfg.noise_params
        data.tr = BurstWithFlowKITTI(root,"train",edition,nframes,noise_info,
                                     crop,resizes,nnf_K,nnf_ps,nnf_exists)
        data.val = BurstWithFlowKITTI(root,"val",edition,nframes,noise_info,
                                      crop,resizes,nnf_K,nnf_ps,nnf_exists)
        data.te = BurstWithFlowKITTI(root,"test",edition,nframes,noise_info,
                                     crop,resizes,nnf_K,nnf_ps,nnf_exists)
    else: raise ValueError(f"Unknown KITTI mode {mode}")
    # for split in data:
    #     print(split,len(data[split]))

    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader

if __name__ == '__main__':
	dataset = read_dataset(resize = (1024, 436))
	# print(dataset['occ'][0].shape)
