# -- python imports --
import os
import cv2
import numpy as np
import numpy.random as npr
from pathlib import Path
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
from datasets.common import get_loader
from datasets.kitti.burst_reader import read_dataset,read_dataset_testing
from datasets.transforms import get_noise_transform
from datasets.kitti.paths import get_kitti_path

class BurstKITTI():

    def __init__(self,root,split,edition,nframes,noise_info,nnf_K,nnf_ps):
        self.root = root
        paths = get_kitti_path(root)
        self.edition = edition
        self.paths = paths
        self.split = split
        self.nframes = nframes
        self.noise_info = noise_info
        self.nnf_K = nnf_K
        self.nnf_ps = nnf_ps
        read_resize = (370, 1224)
        parts = self._get_split_parts_name(split)
        self.dataset = self._read_dataset(paths,edition,parts,nframes,
                                          read_resize,nnf_K,nnf_ps)
        self.noise_xform = get_noise_transform(noise_info)

    def _read_dataset(self,paths,edition,parts,nframes,read_resize,nnf_K,nnf_ps):
        if parts in ["train","val","mixed"]:
            return read_dataset(paths,edition,parts,nframes,
                                resize=read_resize,
                                nnf_K = nnf_K, nnf_ps = nnf_ps)
        elif parts in ["test"]:
            ds = read_dataset_testing(paths,edition,nframes,
                                      resize=read_resize,
                                      nnf_K = nnf_K, nnf_ps = nnf_ps)
            return ds[self.edition]
        else: raise ValueError(f"[KITTI: read_dataset] Uknown part [{parts}]")
            
    def _get_split_parts_name(self,split):
        if split == "trainval": parts = "mixed"
        elif split == "train": parts = "train"
        elif split == "val": parts = "val"
        elif split == "test": parts = "test"
        else: raise ValueError(f"[KITTI: get_split_parts_name] Uknown split [{split}]")
        return parts
        
    def __len__(self):
        return len(self.dataset['image_0'])

    def __getitem__(self,index):
        
        # -- extract images --
        ref_t = torch.FloatTensor(self.dataset['ref_t'][index])
        burst = torch.FloatTensor(self.dataset['burst'][index])
        ref_frame = torch.FloatTensor(self.dataset['ref_frame'][index])
        flows = torch.FloatTensor(self.dataset['flows'][index])
        occs = torch.FloatTensor(self.dataset['occs'][index])
        nnf_vals = torch.FloatTensor(self.dataset['nnf_vals'][index])
        nnf_locs = torch.FloatTensor(self.dataset['nnf_locs'][index])

        # -- apply noise --
        nframes = len(burst)
        clean = [torch.FloatTensor(img) for img in burst]
        noisy = [self.noise_xform(img) for img in burst]
        iid = [self.noise_xform(ref_frame) for l in range(nframes)]

        # -- create burst --
        clean = torch.stack(clean,dim=0)
        noisy = torch.stack(noisy,dim=0)
        iid = torch.stack(iid,dim=0)

        # -- create sample --
        sample = {'clean':clean,'noisy':noisy,'flows':flows,
                  'occs':occs,'iid':iid,'nnf_vals':nnf_vals,
                  'nnf_locs':nnf_locs}
        return sample

def get_burst_kitti_dataset(cfg,mode):
    root = cfg.dataset.root
    root = Path(root)/Path("kitti")
    data = edict()
    batch_size = cfg.batch_size
    rtype = 'dict' if cfg.dataset.dict_loader else 'list'
    nnf_K = 3
    nnf_ps = 3
    if mode == "dynamic":
        edition = "2015"
        nframes = cfg.nframes
        noise_info = cfg.noise_params
        data.tr = BurstKITTI(root,"train",edition,nframes,noise_info,nnf_K,nnf_ps)
        data.val = BurstKITTI(root,"val",edition,nframes,noise_info,nnf_K,nnf_ps)
        data.te = BurstKITTI(root,"test",edition,nframes,noise_info,nnf_K,nnf_ps)
    else: raise ValueError(f"Unknown KITTI mode {mode}")

    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader

if __name__ == '__main__':
	dataset = read_dataset(resize = (1024, 436))
	# print(dataset['occ'][0].shape)
