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
from datasets.kitti.reader import read_dataset,read_dataset_testing
from datasets.transforms import get_noise_transform
from datasets.kitti.paths import get_kitti_path

class KITTI():

    def __init__(self,root,split,edition,nframes,noise_info):
        self.root = root
        paths = get_kitti_path(root)
        self.edition = edition
        self.paths = paths
        self.split = split
        self.nframes = nframes
        self.noise_info = noise_info
        read_resize = (370, 1224)
        parts = self._get_split_parts_name(split)
        self.dataset = self._read_dataset(paths,edition,parts,read_resize)
        self.noise_xform = get_noise_transform(noise_info)

    def _read_dataset(self,paths,edition,parts,read_resize):
        if parts in ["train","val","mixed"]:
            return read_dataset(paths,edition,parts,nframes,resize=read_resize)
        elif parts in ["test"]:
            ds = read_dataset_testing(paths,edition,nframes,resize=read_resize)
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
        img0 = self.dataset['image_0'][index]
        img1 = self.dataset['image_1'][index]
        flow = torch.FloatTensor(self.dataset['flow'][index])
        occ = torch.FloatTensor(self.dataset['occ'][index])
        nnf = torch.FloatTensor(self.dataset['nnf'][index])

        # -- apply noise --
        clean = [torch.FloatTensor(img) for img in [img0,img1]]
        burst = [self.noise_xform(img) for img in [img0,img1]]
        iid = [self.noise_xform(img0) for l in range(len(burst))]

        # -- create burst --
        clean = torch.stack(clean,dim=0)
        burst = torch.stack(burst,dim=0)
        iid = torch.stack(iid,dim=0)

        # -- create sample --
        sample = {'burst':burst,'clean':clean,'flow':flow,'occ':occ,'iid':iid,'nnf':nnf}
        return sample

def get_kitti_dataset(cfg,mode):
    root = cfg.dataset.root
    root = Path(root)/Path("kitti")
    data = edict()
    batch_size = cfg.batch_size
    rtype = 'dict' if cfg.dataset.dict_loader else 'list'
    if mode == "dynamic":
        edition = "2015"
        nframes = cfg.nframes
        noise_info = cfg.noise_params
        data.tr = KITTI(root,"train",edition,nframes,noise_info)
        data.val = KITTI(root,"val",edition,nframes,noise_info)
        data.te = KITTI(root,"test",edition,nframes,noise_info)
    else: raise ValueError(f"Unknown KITTI mode {mode}")

    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader

if __name__ == '__main__':
	dataset = read_dataset(resize = (1024, 436))
	# print(dataset['occ'][0].shape)
