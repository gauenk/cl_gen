# -- python imports --
import numpy as np
from einops import rearrange
from itertools import chain, combinations
from pathlib import Path
import matplotlib.pyplot as plt

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils.misc import images_to_psnrs

def tile_burst_patches(burst,PS,NH):
    N,B,C,H,W = burst.shape
    dilation,padding,stride = 1,0,1
    unfold = nn.Unfold(PS,dilation,padding,stride)
    batch = rearrange(burst,'n b c h w -> (n b) c h w')
    tiled = tile_batch(batch,PS,NH)

    # save_ex = rearrange(tiled,'nb t1 t2 c h w -> nb t1 t2 c h w')
    # save_ex = save_ex[:,NH//2,NH//2,:,:,:]
    # tv_utils.save_image(save_ex,"save_ex.png",normalize=True)

    tiled = rearrange(tiled,'nb t1 t2 c h w -> nb (t1 t2 c) h w')
    patches = unfold(tiled)
    patches = rearrange(patches,'nb (t c ps1 ps2) r -> nb r t ps1 ps2 c',c=C,ps1=PS,ps2=PS)
    patches = rearrange(patches,'(n b) r t ps1 ps2 c -> r b n t c ps1 ps2',n=N)

    # unrolled_img = tile_batches_to_image(patches,NH,PS,H)
    # tv_utils.save_image(unrolled_img,"unroll_img.png",normalize=True)

    return patches
    
def tile_batches_to_image(patches,NH,PS,H):
    REF_NH = NH**2//2 + NH//2*(NH%2==0)
    unrolled_img = patches[:,:,:,REF_NH,:,PS//2,PS//2]
    unrolled_img = rearrange(unrolled_img,'(h w) b n c -> (b n) c h w',h=H)
    return unrolled_img

def tile_batch(batch,PS,NH):
    """
    Creates a tiled version of the input batch to be able to be rolled out using "unfold"
    
    The number of neighborhood pixels to include around each center pixel is "NH"
    The size of the patch around each chosen index (including neighbors) is "PS"
    
    We want to only pad the image once with "reflect". Padding an already padded image
    leads to a "reflection" of a "reflection", and this leads to even stranger boundary 
    condition behavior than whatever one "reflect" will do.

    We extend the image to its final size to apply "unfold" (hence the Hnew, Wnew)
    
    We tile the image w.r.t the neighborhood size so each original center pixel
    has NH neighboring patches.
    """
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
            # print(i+Hnew,j+Wnew,H,W,batch_pad.shape)
            # cmpr = tvF.crop(img_stack[j],PS//2,PS//2,H,W)
            # print("i,j,idx",i,j,idx,images_to_psnrs(batch,cmpr))
            # idx += 1
        img_stack = torch.stack(img_stack, dim=1)
        tiled.append(img_stack)
    tiled = torch.stack(tiled,dim=1)
    return tiled


def tile_bursts_global_motion(burst,PS,NH):
    N = burst.shape[0]
    batch = rearrange(burst,'n b c h w -> (n b) c h w')
    tiled = tile_batch_global_motion(batch,PS,NH)
    burst = rearrange(tiled,'(n b) nh1 nh2 c h w -> b n (nh1 nh2) c h w',n=N)
    return burst
    
def tile_batch_global_motion(batch,PS,NH):
    """
    Creates a tiled version of the input batch to be able to be rolled out using "unfold"
    
    The number of neighborhood pixels to include around each center pixel is "NH"
    The size of the patch around each chosen index (including neighbors) is "PS"
    
    We want to only pad the image once with "reflect". Padding an already padded image
    leads to a "reflection" of a "reflection", and this leads to even stranger boundary 
    condition behavior than whatever one "reflect" will do.

    We extend the image to its final size to apply "unfold" (hence the Hnew, Wnew)
    
    We tile the image w.r.t the neighborhood size so each original center pixel
    has NH neighboring patches.
    """
    B,C,H,W = batch.shape
    batch_pad = F.pad(batch, [NH//2,]*4, mode="reflect")
    tiled,idx = [],0
    for i in range(NH):
        img_stack = []
        for j in range(NH):
            img_stack.append(batch_pad[..., i:i + H, j:j + W])
            # -- test we are correctly cropping --
            # print(i+Hnew,j+Wnew,H,W,batch_pad.shape)
            # cmpr = tvF.crop(img_stack[j],PS//2,PS//2,H,W)
            # print("i,j,idx",i,j,idx,images_to_psnrs(batch,cmpr))
            # idx += 1
        img_stack = torch.stack(img_stack, dim=1)
        tiled.append(img_stack)
    tiled = torch.stack(tiled,dim=1)
    return tiled

