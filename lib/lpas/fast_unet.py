"""
Run a fast unet to single burst
"""

# -- python imports --
import random
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch import --
import torch
from torch import nn
import torch.nn.functional as F

# -- project imports --
from layers.unet import UNet_n2n,UNet_small

def train(cfg,burst,model,optim,nsteps):
    for i in range(nsteps):

        # -- reset --
        model.zero_grad()
        optim.zero_grad()

        # -- rand in and out --
        i,j = random.sample(list(range(burst.shape[0])),2)
        noisy = burst[[i]]
        target = burst[[j]]
        
        # -- forward --
        rec = model(noisy)
        loss = F.mse_loss(rec,target)

        # -- optim step --
        loss.backward()
        optim.step()
    
def test(cfg,burst,model,score_fxn):

    # -- reconstruct a clean image --
    rec = model(burst)+0.5

    # -- compute scores --
    wrapped_score = score_function_wrapper(score_fxn)
    score = wrapped_score(cfg,rec).item()
    return score

def score_function_wrapper(score_fxn):
    def wrapper(cfg,image):
        tmp = image.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scores = score_fxn(cfg,tmp)
        return scores[0,0,0]
    return wrapper

def run_fast_unet(cfg,burst,score_fxn):
    model = UNet_small(3)
    model = model.to(burst.device)
    optim = torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.99))
    train(cfg,burst,model,optim,300)
    score = test(cfg,burst,model,score_fxn)    
    return score
