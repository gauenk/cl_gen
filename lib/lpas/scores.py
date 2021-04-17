
# -- python imports --
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from itertools import chain, combinations

# -- pytorch imports --
import torch
import torch.nn.functional as F

# -- project imports --
from .utils import get_ref_block_index

def get_score_functions(names):
    scores = edict()
    for name in names:
        scores[name] = get_score_function(name)
    return scores

def get_score_function(name):
    if name == "ave":
        return ave_score
    elif name == "got":
        return got_score
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

    ref = repeat(expanded[:,:,:,T//2],'r b e c h w -> r b e tile c h w',tile=T-1)
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta = F.mse_loss(ref,neighbors,reduction='none')
    delta = delta.view(R,B,E,-1)
    delta = torch.mean(delta,dim=3)

    return delta

def refcmp_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    ref = expanded[:,:,:,T//2]
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for t in range(T-1):
        delta_t = F.mse_loss(neighbors[:,:,:,t],ref,reduction='none')
        delta += torch.mean(delta_t.view(R,B,E,-1),dim=3)
    delta /= T
    return delta

def pairwise_delta_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    # ref = repeat(expanded[:,:,:,[T//2]],'r b e c h w -> r b e tile c h w',tile=T-1)
    # neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for t1 in range(T):
        for t2 in range(T):
            delta_t = F.mse_loss(expanded[:,:,:,t1],expanded[:,:,:,t2],reduction='none')
            delta += torch.mean(delta_t.view(R,B,E,-1),dim=3)
    delta /= T*T
    return delta

#
# Grid Functions
#

# -- run over the grids for below --
def delta_over_grids(cfg,expanded,grids):
    R,B,E,T,C,H,W = expanded.shape
    delta = torch.zeros(R,B,E,device=expanded.device)
    for set0,set1 in grids:
        set0,set1 = np.atleast_1d(set0),np.atleast_1d(set1)
        ave0 = torch.mean(expanded[:,:,:,set0],dim=3)
        ave1 = torch.mean(expanded[:,:,:,set1],dim=3)
        delta_t = F.mse_loss(ave0,ave1,reduction='none').view(R,B,E,-1)
        delta += torch.mean(delta_t,dim=3)
    delta /= len(grids)
    return delta

def powerset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- powerset --
    indices = np.arange(T)
    powerset = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T))
    powerset = np.array([np.array(elem) for elem in list(powerset)])
    grids = np.array(np.meshgrid(*[powerset,powerset]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta = delta_over_grids(cfg,expanded,grids)
    return delta

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
    delta = delta_over_grids(cfg,expanded,grids)
    return delta

def lgsubset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- compare large subsets --
    indices = np.arange(T)
    subset_lg = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T-2,T))
    subset_lg = np.array([np.array(elem) for elem in list(subset_lg)])
    grids = np.array(np.meshgrid(*[subset_lg,subset_lg]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta = delta_over_grids(cfg,expanded,grids)
    return delta


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
    delta = delta_over_grids(cfg,expanded,grids)
    return delta

def lgsubset_v_ref_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    subset_lg = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T-2,T))
    subset_lg = np.array([np.array(elem) for elem in list(subset_lg)])
    grids = np.array(np.meshgrid(*[subset_lg,[[T//2,]]]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta = delta_over_grids(cfg,expanded,grids)
    return delta

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
    delta = delta_over_grids(cfg,expanded,grids)
    return delta

def powerset_v_ref_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    powerset = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T))
    powerset = np.array([np.array(elem) for elem in list(powerset)])
    grids = np.array(np.meshgrid(*[powerset,[T//2]]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta = delta_over_grids(cfg,expanded,grids)
    return delta


def create_n_grids(T):
    # -- create init indices --
    indices = np.arange(T)
    I = indices.shape[0]

    # -- powerset --
    powerset = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(I))
    powerset = np.array([np.array(elem) for elem in list(powerset)])

    # -- subset --
    subset_lg = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(I-2,I))
    subset_lg = np.array([np.array(elem) for elem in list(subset_lg)])

    subset_sm = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(0,2))
    subset_sm = np.array([np.array(elem) for elem in list(subset_sm)])

    subset_ex = np.r_[subset_sm,subset_lg]

    # -- indices appended with group --
    l_indices = [[i] for i in indices]
    l_indices.append(list(np.arange(T)))

    # -- create grid --

    # grids = np.array(np.meshgrid(*[l_indices,l_indices]))
    # grids = np.array(np.meshgrid(*[indices,indices]))
    # grids = np.array(np.meshgrid(*[subset_ex,subset_ex]))
    grids = np.array(np.meshgrid(*[subset_lg,subset_lg]))
    # grids = np.array(np.meshgrid(*[subset_lg,subset_sm]))
    # grids = np.array(np.meshgrid(*[subset_sm,subset_sm]))
    # grids = np.array(np.meshgrid(*[powerset,subset_lg]))
    # grids = np.array(np.meshgrid(*[powerset,indices]))
    # grids = np.array(np.meshgrid(*[powerset,powerset]))
    # grids = np.array(np.meshgrid(*[powerset,[0]]))
    # grids = np.array(np.meshgrid(*[subset_lg,[0]]))
    # grids = np.array(np.meshgrid(*[subset_sm,[0]]))
    # grids = np.array(np.meshgrid(*[ [[0,1],[0,2],[1,2]] ,[0] ]))
    # grids = np.array(np.meshgrid(*[ [[0,1],[0,2],[0,1,2]] ,[[0]] ]))
    # grids = np.array(np.meshgrid(*[ [[2],] ,[[0]] ]))
    grids = np.array([grid.ravel() for grid in grids]).T
    return grids

def largesmallset_score(cfg,expanded):
    pass


def got_score(cfg,expanded):
    pass

def emd_score(cfg,expanded):
    pass

