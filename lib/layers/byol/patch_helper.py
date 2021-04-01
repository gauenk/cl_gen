# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

class PatchHelper():

    def __init__(self, num_frames, patchsize, nh_size, img_size):
        
        # -- init vars --
        self.num_frames = num_frames
        self._patchsize = patchsize
        self.nh_size = nh_size # number of patches around center pixel
        self.img_size = img_size

        # -- unfold input patches --
        padding,stride,ipad = 1,1,self.ps//2
        self.unfold_input = nn.Unfold(self._patchsize,1,padding,stride)

        # -- create grid to compute indice --
        index_grid = torch.arange(0,img_size**2).reshape(1,1,img_size,img_size)
        self.index_grid = F.pad(index_grid.type(torch.float),
                                (ipad,ipad,ipad,ipad),mode='reflect')[0,0].type(torch.long)
        self.index_pad = self.index_grid.shape[0] - self.img_size
    
        # -- indexing bursts --
        self.midx = num_frames // num_frames if num_frames != 2 else 1
        self.no_mid_idx = np.r_[np.r_[:self.midx],np.r_[self.midx+1:num_frames]]
        self.no_mid_idx = torch.LongTensor(self.no_mid_idx)

        
    @property
    def ps(self):
        return self._patchsize

    def gather_local_patches(self, burst, in_index):
        ps,nh,N = self.ps,self.nh_size,len(self.no_mid_idx)
        window = self.index_window(in_index,nh)
        local_patches = []
        for neighbor_index in window:
            h_window,w_window = self.hw_window(neighbor_index,ps)
            patch = burst[:,:,:,h_window,w_window]
            patch = rearrange(patch,'n b c (h w) -> n b c h w',h=ps)
            local_patches.append(patch)
        local_patches = torch.stack(local_patches,dim=2)
        return local_patches

    def hw_window(self,index,ps=None):
        if ps is None: ps = self.nh_size
        I = self.img_size
        window = self.index_window(index,ps)
        h_window = window // I
        w_window = window % I
        return h_window,w_window

    def index_window(self,index,ps=None):
        if ps is None: ps = self.ps
        I,ipad = self.img_size,self.index_pad
        row = index // I + ipad
        col = index % I + ipad
        top,left = row - ps//2,col - ps//2
        index_window = tvF.crop(self.index_grid,top,left,ps,ps).reshape(-1)
        return index_window

    def apply_to_burst(self,model,burst):
        if burst.shape[0] > 2: print("WARNING: not using N > 2 right now")
        ps = self.ps
        patches = self.unfold(burst) # burst.shape = (N,C,H,W)
        patches = rearrange(patches,'n b l c h w -> (n b l) c h w')
        patches = rearrange(patches,'n (c ps1 ps2) r -> n r c ps1 ps2',ps1=ps,ps2=ps)
        features = model(patches,return_embedding=True)
        features = rearrange(features,'n (c ps1 ps2) r -> n r c ps1 ps2',ps1=ps,ps2=ps)
        features = rearrange(features,'(h w) f -> f h w',b=B,n=N,h=H)
        return features

    def apply_to_image(self,image):
        ps = self.ps
        patches = self.unfold(rearrange(image,'c h w -> 1 c h w'))
        patches = rearrange(patches,'1 (c ps1 ps2) r -> 1 r c ps1 ps2',ps1=ps,ps2=ps)
        features = self(patches)
        features = rearrange(features,'(h w) f -> f h w',b=B,n=N,h=H)
        return features
