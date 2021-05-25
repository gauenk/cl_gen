
import numpy as np
import pickle,sys,os,yaml,io
from einops import rearrange,repeat
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F

def create_sobel_filter():
    # -- get sobel filter to detect rough spots --
    sobel = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_t = sobel.t()
    sobel = sobel.reshape(1,3,3)
    sobel_t = sobel_t.reshape(1,3,3)
    weights = torch.stack([sobel,sobel_t],dim=0)
    return weights

def apply_sobel_filter(image):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    C = image.shape[-3]
    weights = create_sobel_filter()
    weights = weights.to(image.device)
    weights = repeat(weights,'b 1 h w -> b c h w',c=C)
    edges = F.conv2d(image,weights,padding=1,stride=1)
    edges = ( edges[:,0]**2 + edges[:,1]**2 ) ** (0.5)
    return edges
