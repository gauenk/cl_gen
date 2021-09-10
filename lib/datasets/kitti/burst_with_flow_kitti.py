# -- python imports --
import os
import cv2
import numpy as np
import numpy.random as npr
from pathlib import Path
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
from datasets.common import get_loader
from datasets.kitti.burst_with_flow_reader import read_dataset_paths,read_dataset_testing,read_dataset_sample
from datasets.transforms import get_noise_transform
from datasets.kitti.paths import get_kitti_path

class BurstWithFlowKITTI():

    def __init__(self,root,split,edition,nframes,noise_info,nnf_K,nnf_ps):
        self.root = root
        paths = get_kitti_path(root)
        self.edition = edition
        self.paths = paths
        self.split = split
        self.istest = split == "test"
        self.nframes = nframes
        self.noise_info = noise_info
        self.nnf_K = nnf_K
        self.nnf_ps = nnf_ps
        self.read_resize = (370, 1224)
        parts = self._get_split_parts_name(split)
        self.dataset = self._read_dataset_paths(paths,edition,parts,nframes,
                                                self.read_resize,nnf_K,nnf_ps)
        self.noise_xform = get_noise_transform(noise_info,use_to_tensor=False)

    def _read_dataset_paths(self,paths,edition,parts,nframes,read_resize,nnf_K,nnf_ps):
        if parts in ["train","val","mixed"]:
            return read_dataset_paths(paths,edition,parts,nframes,
                                      resize=read_resize,
                                      nnf_K = nnf_K, nnf_ps = nnf_ps)
        elif parts in ["test"]:
            return read_dataset_testing(paths,edition,nframes,
                                        resize=read_resize,
                                        nnf_K = nnf_K, nnf_ps = nnf_ps)
        else: raise ValueError(f"[KITTI: read_dataset] Uknown part [{parts}]")
            
    def _get_split_parts_name(self,split):
        if split == "trainval": parts = "mixed"
        elif split == "train": parts = "train"
        elif split == "val": parts = "val"
        elif split == "test": parts = "test"
        else: raise ValueError(f"[KITTI: get_split_parts_name] Uknown split [{split}]")
        return parts
        
    def __len__(self):
        return len(self.dataset['burst_id'])

    def __getitem__(self,index):
        
        # -- read sample --
        burst_id = self.dataset['burst_id'][index]
        ref_t = self.dataset['ref_t'][index]
        edition = self.dataset['edition'][index]
        data = read_dataset_sample(burst_id,ref_t,self.nframes,edition,
                                   self.istest,path=self.paths,
                                   resize = self.read_resize,
                                   nnf_K = self.nnf_K, nnf_ps = self.nnf_ps)

        # -- extract data --
        burst = torch.FloatTensor(data['burst'])
        ref_frame = tvF.to_tensor(data['ref_frame'])
        flows = torch.FloatTensor(data['flows'])
        occs = torch.FloatTensor(data['occs'])
        nnf_vals = torch.FloatTensor(data['nnf_vals'])
        nnf_locs = torch.FloatTensor(data['nnf_locs'])

        # -- rename clean frames --
        nframes = len(burst)
        clean = burst

        # -- create noisy sample --
        noisy = [self.noise_xform(img) for img in burst]
        noisy = torch.stack(noisy,dim=0)

        # -- create static iid noisy samples --
        iid = [self.noise_xform(ref_frame) for l in range(nframes)]
        iid = torch.stack(iid,dim=0)

        # -- create sample --
        sample = {'clean':clean,'noisy':noisy,'flows':flows,
                  'occs':occs,'iid':iid,'nnf_vals':nnf_vals,
                  'nnf_locs':nnf_locs,'index':index}
        return sample

def get_burst_with_flow_kitti_dataset(cfg,mode):
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
        data.tr = BurstWithFlowKITTI(root,"train",edition,nframes,noise_info,nnf_K,nnf_ps)
        data.val = BurstWithFlowKITTI(root,"val",edition,nframes,noise_info,nnf_K,nnf_ps)
        data.te = BurstWithFlowKITTI(root,"test",edition,nframes,noise_info,nnf_K,nnf_ps)
    else: raise ValueError(f"Unknown KITTI mode {mode}")

    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader

if __name__ == '__main__':
	dataset = read_dataset(resize = (1024, 436))
	# print(dataset['occ'][0].shape)
