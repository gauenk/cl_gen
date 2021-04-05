
# -- python imports --
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from joblib import Parallel, delayed
from functools import partial

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import torchvision.utils as tv_utils

# -- project imports --
from layers.unet import UNet_small
from .byol_pytorch import MLP
from pyutils.misc import images_to_psnrs


class UNetBYOL(nn.Module):

    def __init__(self, num_frames, num_in_ftrs, num_out_ftrs, num_out_channels,
                 patchsize, nh_size, img_size, attn_params = None):
        super().__init__()

        # -- init --
        self.num_frames = num_frames
        self.num_in_ftrs = num_in_ftrs
        self.num_out_ftrs = num_out_ftrs
        self.nh_size = nh_size # number of patches around center pixel
        self._patchsize = patchsize
        self.img_size = img_size
        self.layer_norm = nn.LayerNorm(num_out_channels*patchsize**2)

        # -- create model --
        self.unet = UNet_small(num_in_ftrs+3,num_out_channels)
        # self.unet_out_size = num_out_channels * patchsize**2
        patchsize_e = patchsize + (patchsize % 2) # add one dim for odd input size
        self.unet_out_size = num_out_channels * patchsize_e**2
        self.mlp = MLP( self.unet_out_size, num_out_ftrs, hidden_size = 1024)

    def forward(self,input_patches):
        """
        params:
           [patches] shape: (B*L,C,H,W)
        """

        # -- init shapes --
        BL,C,H,W = input_patches.shape
        L = self.nh_size**2 + 1 # all my neighbors + myself
        B = BL // L

        # -- averaging with ref --
        patches = rearrange(input_patches,'(b l) c h w -> b l c h w',b=B)

        # print("unet_patches.shape",patches.shape)
        rep_ref_patch = repeat(patches[:,0],'b c h w -> b rep c h w',rep=L-1)
        ave_patches = ( rep_ref_patch + patches[:,1:] ) / 2.
        ave_patches = torch.cat([patches[:,[0]],ave_patches],dim=1)
        ave_patches = rearrange(ave_patches,'b l c h w -> b (l c) h w',b=B)
        ave_patches = F.pad(ave_patches,[1,0,1,0]) # -- unet must be 2^K input dim

        # -- standard foward pass --
        # patches = rearrange(patches,'(b l) c h w -> b (l c) h w',b=B)
        unet_embeddings = self.unet(ave_patches)
        # emb_sum = (unet_embeddings + patches[:,0]).reshape(B,-1)
        # unet_embeddings = self.layer_norm( emb_sum )
        # unet_embeddings = self.layer_norm( unet_embeddings.reshape(B,-1) )
        unet_embeddings = unet_embeddings.reshape(B,-1)
        embeddings = self.mlp( unet_embeddings )
        #embeddings = self.mlp(unet_embeddings.reshape(B,-1))

        # -- old method foward pass --
        # patches = rearrange(patches,'(b l) c h w -> b (l c) h w',b=B)
        # unet_embeddings = self.unet(patches).reshape(B,-1)
        # embeddings = self.mlp(unet_embeddings)

        # -- [testing patch recon] spoof no xform --
        """
        Still see a gap even with this! 43.66 v 49.28
        """
        # crop = rearrange(tvF.crop(input_patches,7,7,3,3),'b c h w -> b (c h w)')
        # crop = rearrange(tvF.crop(input_patches,8,8,3,3),'b c h w -> b (c h w)')
        # a,b,c = patches.min().item(),patches.max().item(),patches.mean().item()
        # print("[unet] ","%2.2f, %2.2f, %2.2f" % (a,b,c) )
        # crop = patches[:,0,:,8,8]
        # pad_len = 32 - crop.shape[1]
        # crop = F.pad(crop,[0,pad_len])
        # print(crop[0,:3],crop[0,-3:])
        # print("pad",crop.shape)
        # return crop
        return embeddings


# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

class UNetPatchHelper():

    def __init__(self, num_frames, patchsize, nh_size, img_size):
        
        # -- init vars --
        self.num_frames = num_frames
        self._patchsize = patchsize
        self.nh_size = nh_size # number of patches around center pixel
        self.img_size = img_size

        # -- unfold input patches --
        padding,stride,ps_ipad,nh_ipad = 0,1,self.ps//2,self.nh_size//2
        self.unfold_input = nn.Unfold(self._patchsize,1,padding,stride)

        # -- create grid to compute indice --
        index_grid = torch.arange(0,img_size**2).reshape(1,1,img_size,img_size)
        self.index_grid,self.index_pad = {},{}
        for ipad in [1,ps_ipad,nh_ipad]:
            ipad_s = str(ipad)
            padding = (ipad,ipad,ipad,ipad)
            index_grid_padded = F.pad(index_grid.type(torch.float),padding,mode='reflect')
            index_grid_padded = index_grid_padded[0,0].type(torch.long)
            self.index_grid[ipad_s] = index_grid_padded
            self.index_pad[ipad_s] = (index_grid_padded.shape[0] - self.img_size)//2
    
        # -- indexing bursts --
        self.midx = num_frames // num_frames if num_frames != 2 else 1
        self.no_mid_idx = np.r_[np.r_[:self.midx],np.r_[self.midx+1:num_frames]]
        self.no_mid_idx = torch.LongTensor(self.no_mid_idx)

        
    @property
    def ps(self):
        return self._patchsize

    def pad_burst(self, burst):
        ipad = self.ps //2
        N = burst.shape[0]
        burstNB = rearrange(burst,'n b c h w -> (n b) c h w')
        burst_pad = F.pad(burstNB,(ipad,ipad,ipad,ipad),mode='reflect')
        burst = rearrange(burst_pad,'(n b) c h w -> n b c h w',n=N)
        return burst

    def form_input_patches(self,local_patches):
        N,B,L,C,H,W = local_patches.shape
        mid_idx = L // 2

        # -- use local information from neighbor images --
        ref_patch = local_patches[0][:,[mid_idx]]
        nh_patches = local_patches[1]
        inputs = torch.cat([ref_patch,nh_patches],dim=1)
        inputs = rearrange(inputs,'b l c h w -> (b l) c h w')

        # -- initial results for single patch only --
        # inputs = repeat(ref_patch,'b l c h w -> b (tile l) c h w',tile=L+1)
        # inputs = repeat(inputs,'b l c h w -> (b l) c h w')
        return inputs

    def embeddings_to_image(self,embeddings):
        I  = self.img_size
        ftr_img = rearrange(embeddings,'(h w b) f -> 1 b f h w',h=I,w=I)
        return ftr_img

    def tile_batch(self,batch,PS,NH):
        B,C,H,W = batch.shape
        Hnew,Wnew = H + 2*(PS//2),W + 2*(PS//2)
        M = PS//2 + NH//2 # reaching up NH/2 center-pixels. Then reach up PS/2 more pixels
        batch_pad = F.pad(batch, [M,]*4, mode="reflect")
        tiled,idx = [],0
        for i in range(NH):
            img_stack = []
            for j in range(NH):
                img_stack.append(batch_pad[..., i:i + Hnew, j:j + Wnew])
                # -- test we are correctly cropping --
                # cmpr = tvF.crop(img_stack[j],PS//2,PS//2,H,W)
                # print("i,j,idx",i,j,idx,images_to_psnrs(batch,cmpr))
                # idx += 1
            img_stack = torch.stack(img_stack, dim=1)
            tiled.append(img_stack)
        tiled = torch.stack(tiled,dim=1)
        return tiled

    def prepare_burst_patches(self,burst):
        """
        burst[0] == src image to be converted to ftrs
        burst[1] == another dynamic frame

        burst.shape = (N,B,C,H,W)

        """
        if burst.shape[0] > 2: print("WARNING: not using N > 2 right now")
        # from joblib import Parallel, delayed
        # >>> Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
        # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        N,B,C,H,W = burst.shape
        burstNB = rearrange(burst,'n b c h w -> (n b) c h w')

        # -- get tiled images; only add padding once (else we "reflect" a "reflect". ew.) --
        PS,NH = self.ps,self.nh_size
        tiled = rearrange(self.tile_batch(burstNB,PS,NH),'nb t1 t2 c h w -> nb (t1 t2 c) h w')
        patches = self.unfold_input(tiled)
        patches = rearrange(patches,'nb (t c ps1 ps2) r -> nb r t ps1 ps2 c',c=C,ps1=PS,ps2=PS)

        # -- parallel code --
        # patches = [None for i in range(HW)]
        # def parallel_proc(self,patches,index):
        #     patches[index] = self.gather_local_patches(burst, index)
        # pp_setup = partial(parallel_proc,self,patches)
        # Parallel(n_jobs=8,prefer="threads")(delayed(pp_setup)(index) for index in range(HW))

        # -- serial code --
        patches = []
        for index in range(self.img_size**2):
            patches_i = self.gather_local_patches(burst, index)
            patches.append(patches_i)

        patches = torch.stack(patches,dim=0)
        print("p",patches.shape)
        return patches

    def gather_local_patches(self, burst, in_index):
        # burst_pad = self.pad_burst(burst)
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
        # -- indexing with padding --
        if ps is None: ps = self.ps
        ps_s = str(ps//2)
        index_grid,ipad = self.index_grid[ps_s],self.index_pad[ps_s]

        # -- image row and col --
        I = self.img_size
        row = index // I + ipad
        col = index % I + ipad
        top,left = row - ps//2,col - ps//2
        index_window = tvF.crop(index_grid,top,left,ps,ps).reshape(-1)
        return index_window

    def index_to_hw(self,index):
        row = index // self.img_size
        col = index % self.img_size
        return row, col

    def apply_to_burst(self,model,burst):
        """
        burst[0] == src image to be converted to ftrs
        burst[1] == another dynamic frame
        """
        if burst.shape[0] > 2: print("WARNING: not using N > 2 right now")
        ps = self.ps
        img_ftrs = torch.zeros(ps,self.img_size,self.img_size)
        for index in range(self.img_size**2):
            patches = self.gather_local_patches(burst, index)
            patches = rearrange(patches,'n b l c h w -> (n b l) c h w') # for attn
            features = model(patches,return_embedding=True)
            features = rearrange(features,'(b l) f -> b l f')
            features = torch.mean(features,dim=1) 
            r,c = self.index_to_hw(index)
            img_ftrs[:,r,c] = features
        return img_ftrs

    def apply_to_image(self,image):
        ps = self.ps
        patches = self.unfold(rearrange(image,'c h w -> 1 c h w'))
        patches = rearrange(patches,'1 (c ps1 ps2) r -> 1 r c ps1 ps2',ps1=ps,ps2=ps)
        features = self(patches)
        features = rearrange(features,'(h w) f -> f h w',b=B,n=N,h=H)
        return features


    
