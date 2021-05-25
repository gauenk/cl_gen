"""
Run experiments to set the nh and ps size given a fixed ppf dynamic

"""

# -- python imports --
import bm3d
import asyncio
import pandas as pd
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange, repeat, reduce
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

# -- project code --
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr,images_to_psnrs

def test_ps_nh_sizes(cfg,model,burst,n_indices=2):
    B = np.min([burst.shape[1],3])
    indices = np.random.choice(cfg.frame_size**2,n_indices)
    patches = []
    for index in indices:
        print(model.patch_helper.index_to_hw(index),index)
        index_window = model.patch_helper.index_window(index,ps=3)
        for nh_index in index_window:
            patches_i = model.patch_helper.gather_local_patches(burst+0.5, nh_index)
            patches.append(patches_i)
    patches = torch.cat(patches,dim=1)
    input_patches = model.patch_helper.form_input_patches(patches)

    L = cfg.byol_nh_size**2 + 1
    patches = rearrange(input_patches,'(b l) c h w -> b l c h w',l=L)

    for b in range(B):
        print(b)
        burstN = rearrange(burst[:,[b]],'n b c h w -> (n b) c h w')
        tv_utils.save_image(burstN,f"burst_{b}.png",normalize=True)
        tgt = repeat(patches[b,[0]],'1 c h w -> tile c h w ',tile=L-1)
        guess = patches[b,1:]
        print("max_psnrs:",np.max(images_to_psnrs(tgt,guess)))
        tv_utils.save_image(patches[b],f"patches_{b}.png",normalize=True)
        delta = tgt - guess
        tv_utils.save_image(delta,f"delta_{b}.png",normalize=True)

    
