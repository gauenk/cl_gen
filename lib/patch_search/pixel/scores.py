
# -- python imports --
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from itertools import chain, combinations

# -- pytorch imports --
import torch
import torch.nn.functional as F

# -- project imports --
from layers.unet import UNet_n2n,UNet_small

# -- [local] project imports --
from ..utils import get_ref_block_index

def get_score_functions(names):
    scores = edict()
    for name in names:
        scores[name] = get_score_function(name)
    return scores

def get_score_function(name):
    if name == "ave":
        return ave_score
    elif name == "gaussian_ot":
        return gaussian_ot_score
    elif name == "emd":
        return emd_score        
    elif name == "powerset":
        return powerset_score
    elif name == "extrema":
        return extrema_score
    elif name == "lgsubset":
        return lgsubset_score
    elif name == "lgsubset_v_indices":
        return lgsubset_v_indices_score
    elif name == "lgsubset_v_ref":
        return lgsubset_v_ref_score
    elif name == "powerset_v_indices":
        return powerset_v_indices_score
    elif name == "powerset_v_ref_score":
        return powerset_v_ref_score
    elif name == "pairwise":
        return pairwise_delta_score
    elif name == "refcmp":
        return refcmp_score
    else:
        raise ValueError(f"Uknown score function [{name}]")

def ave_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    ref = repeat(expanded[:,:,:,T//2],'r b e c h w -> r b e tile c h w',tile=T-1)
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta = F.mse_loss(ref,neighbors,reduction='none')
    delta = delta.view(R,B,E,T-1,-1)
    delta_t = torch.mean(delta,dim=4)
    delta = torch.mean(delta_t,dim=3)

    # -- append dim for T --
    Tm1 = T-1
    zeros = torch.zeros_like(delta_t[:,:,:,[0]])
    delta_t = torch.cat([delta_t[:,:,:,:Tm1//2],zeros,delta_t[:,:,:,Tm1//2:]],dim=3)

    return delta,delta_t

def refcmp_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    ref = expanded[:,:,:,T//2]
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta_t = torch.zeros(R,B,E,T,device=expanded.device)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for t in range(T-1):
        delta_pair = F.mse_loss(neighbors[:,:,:,t],ref,reduction='none')
        delta_t += delta_pair
        delta += torch.mean(delta_t.view(R,B,E,-1),dim=3)
    delta_t /= (T-1)
    delta /= (T-1)
    return delta,delta_t

def pairwise_delta_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    # ref = repeat(expanded[:,:,:,[T//2]],'r b e c h w -> r b e tile c h w',tile=T-1)
    # neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta_t = torch.zeros(R,B,E,T,device=expanded.device)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for t1 in range(T):
        for t2 in range(T):
            delta_pair = F.mse_loss(expanded[:,:,:,t1],expanded[:,:,:,t2],reduction='none')
            delta_t[:,:,:,[t1,t2]] += delta_pair
            delta += torch.mean(delta_t.view(R,B,E,-1),dim=3)
    delta /= T*T
    delta_t /= T*T
    return delta,delta_t

#
# Grid Functions
#

# -- run over the grids for below --
def delta_over_grids(cfg,expanded,grids):
    R,B,E,T,C,H,W = expanded.shape
    unrolled = rearrange(expanded,'r b e t c h w -> r b e t (c h w)')
    delta_t = torch.zeros(R,B,E,T,device=expanded.device)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for set0,set1 in grids:
        set0,set1 = np.atleast_1d(set0),np.atleast_1d(set1)

        # -- compute ave --
        ave0 = torch.mean(expanded[:,:,:,set0],dim=3)
        ave1 = torch.mean(expanded[:,:,:,set1],dim=3)

        # -- rearrange --
        ave0 = rearrange(ave0,'r b e c h w -> r b e (c h w)')
        ave1 = rearrange(ave1,'r b e c h w -> r b e (c h w)')

        # -- rep across time --
        ave0_repT = repeat(ave0,'r b e f -> r b e t f',t=T)
        ave1_repT = repeat(ave1,'r b e f -> r b e t f',t=T)

        # -- compute deltas --
        delta_pair = F.mse_loss(ave0,ave1,reduction='none').view(R,B,E,-1)
        delta_0 = F.mse_loss(ave0_repT,unrolled,reduction='none').view(R,B,E,T,-1)
        delta_1 = F.mse_loss(ave1_repT,unrolled,reduction='none').view(R,B,E,T,-1)
        delta_t += torch.mean( (delta_0 + delta_1)/2., dim = 4)
        delta += torch.mean(delta_pair,dim=3)
    delta /= len(grids)
    delta_t /= len(grids)
    return delta,delta_t

def powerset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- powerset --
    indices = np.arange(T)
    powerset = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T))
    powerset = np.array([np.array(elem) for elem in list(powerset)])
    grids = np.array(np.meshgrid(*[powerset,powerset]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def extrema_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- extrema subsets --
    indices = np.arange(T)
    subset_lg = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T-2,T))
    subset_lg = np.array([np.array(elem) for elem in list(subset_lg)])
    subset_sm = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(0,2))
    subset_sm = np.array([np.array(elem) for elem in list(subset_sm)])
    subset_ex = np.r_[subset_sm,subset_lg]
    grids = np.array(np.meshgrid(*[subset_ex,subset_ex]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def lgsubset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- compare large subsets --
    indices = np.arange(T)
    subset_lg = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T-2,T))
    subset_lg = np.array([np.array(elem) for elem in list(subset_lg)])
    grids = np.array(np.meshgrid(*[subset_lg,subset_lg]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t


def lgsubset_v_indices_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    l_indices = [[i] for i in indices]
    l_indices.append(list(np.arange(T)))
    subset_lg = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T-2,T))
    subset_lg = np.array([np.array(elem) for elem in list(subset_lg)])
    grids = np.array(np.meshgrid(*[subset_lg,l_indices]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def lgsubset_v_ref_score(cfg,expanded,ref_t=None):
    R,B,E,T,C,H,W = expanded.shape
    if ref_t is None: ref_t = T//2

    # -- indices and large subset --
    indices = np.arange(T)
    subset_lg = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T-2,T))
    subset_lg = np.array([np.array(elem) for elem in list(subset_lg)])
    grids = np.array(np.meshgrid(*[subset_lg,[[ref_t,]]]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def powerset_v_indices_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    l_indices = [[i] for i in indices]
    l_indices.append(list(np.arange(T)))
    powerset = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T))
    powerset = np.array([np.array(elem) for elem in list(powerset)])
    grids = np.array(np.meshgrid(*[powerset,l_indices]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def powerset_v_ref_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    powerset = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T))
    powerset = np.array([np.array(elem) for elem in list(powerset)])
    grids = np.array(np.meshgrid(*[powerset,[T//2]]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

#
# Optimal Transport Based Losses
# 

def gaussian_ot_score(cfg,expanded,return_frames=False):
    R,B,E,T,C,H,W = expanded.shape
    vectorize = rearrange(expanded,'r b e t c h w -> (r b e t) (c h w)')
    means = torch.mean(vectorize,dim=1)
    stds = torch.std(vectorize,dim=1)

    # -- gaussian zero mean, var = noise_level --
    gt_std = cfg.noise_params['g']['stddev']/255.
    loss = means**2
    loss += (stds**2 - 2*gt_std**2)**2
    losses_t = rearrange(loss,'(r b e t) -> r b e t',r=R,b=B,e=E,t=T)
    losses = torch.mean(losses_t,dim=3)
    return losses,losses_t

def emd_score(cfg,expanded):
    pass

