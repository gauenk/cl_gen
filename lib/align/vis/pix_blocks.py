

# -- python imports --
import numpy as np
import numpy.random as npr
from einops import rearrange

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
import align.combo_align._block_utils as block_utils


def create_patch(T,H,PS):

    image = torch.Tensor(npr.rand((T,1,3,50,50)))
    center = tvF.center_crop(path[0],PS,PS)
    patches = tile_across_blocks(image,H)
    return image,center,patches


def run():

    T = 3
    H = 3
    PS = 5
    image,center,patches = create_patches(T,H,PS)
    
    
