
# -- python imports --
import sys,os,random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch import --
import torch
from torch import nn
import torchvision.utils as tv_utils
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as tvF

class RecordingHooks():

    def __init__(self):
        self.activations = []

    def __call__(self, module, inputs, outputs):
        # reshaped_filter = rearrange(output_filter,'b (n k2) h w -> b n k2 1 h w',n=self.N)
        self.activations.append(outputs)
        
    def clear(self):
        self.activations = []

def activation_trace(model,burst,fxn_str):
    T = burst.shape[0]
    record_hook = RecordingHooks()
    hooks = []
    for name,module in model.named_modules():
        hooks.append(module.register_forward_hook(record_hook))
    rec = model(burst)
    for hook in hooks: hook.remove()

    A = len(record_hook.activations)
    loss = 0
    for activation in record_hook.activations:
        ref = activation[T//2]
        for t in range(T):
            if t == T//2: continue
            mse = F.mse_loss(ref,activation[t],reduction='none')
            wmse = mse * torch.abs(ref)
            loss += torch.mean(wmse)
            # loss += F.mse_loss(ref,activation[t]).item()
    loss /= (T * A)
    return loss
