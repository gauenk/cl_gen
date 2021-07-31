
# -- python imports --
import math
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from itertools import chain, combinations
from pathlib import Path
from functools import partial

# -- pytorch imports --
import torch
import torch.nn.functional as F

# -- project imports --
from pyutils import torch_xcorr,create_combination,print_tensor_stats,save_image,create_subset_grids,create_subset_grids,create_subset_grids_fixed,ncr
from layers.unet import UNet_n2n,UNet_small
from layers.ot_pytorch import sink_stabilized,pairwise_distances,dmat


def get_score_functions(names):
    scores = edict()
    for name in names:
        scores[name] = get_score_function(name)
    return scores

def get_score_function(name):
    if name.split("-")[0] == "flownet":
        return get_flownet_function(name)
    else:
        raise ValueError(f"Uknown score function [{name}]")

# def get_flownet_function(name):
#     flownet,version,noise_type,noise_level = name.split("-")
#     if version != "v2": raise ValueError(f"Unknown flownet version [{version}]")
#     model_path = FLOWNET_MODEL_PATH / version / noise_type / noise_level
#     print(f"Loading flownet model from [{model_path}]")
#     model = load_flownet_model(version,model_path,default_gpuid)
#     registered_flownet = partial(compute_flownet_flow,model)
#     return registered_flownet


def compute_flownet_of(model,cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    device = expanded.device
    model = model.to(device)

    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2

    # -- shape to pairs --
    ref = repeat(samples[T//2],'d be -> tile d be',tile=T-1)
    neighbors = torch.cat([samples[:T//2],samples[T//2+1:]],dim=0)
    pairs = torch.stack([ref,neighbors],dim=0) # 2, T-1, D, BE
    pairs = rearrange(pairs,'two tm1 d be -> (tm1 be) two d')

    # -- compute flow --
    flow = model(pairs)
    
    

    
    flows = model(samples)
    
