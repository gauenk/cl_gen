# -=-=-=-=-=-=-=-=-
#
#    ABPS Utils
# 
# -=-=-=-=-=-=-=-=-

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

def aligned_burst_from_indices_global_dynamics(patches,n_indices,nh_indices):
    # new: 4:02, added n_indices
    R,B = patches.shape[:2]
    N = n_indices.shape[0]
    best_patches = []
    for b in range(B):
        patches_n = []
        for n in range(N):
            patches_n.append(patches[:,b,n_indices[n],nh_indices[b,n],:,:,:])
        patches_n = torch.stack(patches_n,dim=1)
        best_patches.append(patches_n)
    best_patches = torch.stack(best_patches,dim=1)
    return best_patches

def aligned_burst_image_from_indices_global_dynamics(patches,n_indices,nh_indices):
    aligned_patches = aligned_burst_from_indices_global_dynamics(patches,n_indices,nh_indices)
    R,PS = aligned_patches.shape[0],aligned_patches.shape[-1]
    H = int(np.sqrt(R))
    aligned = rearrange(aligned_patches[:,:,:,:,PS//2,PS//2],'(h w) b n c -> n b c h w',h=H)
    return aligned

def create_split_burst_grid(N):

    # -- adjust for ref patch --
    indices = np.arange(N-1)
    indices[N//2]+=2
    I = indices.shape[0]

    # -- sets with many elements --
    splits = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(I-3,I-2))
    splits = np.array([np.array(elem) for elem in list(splits)])

    # -- shuffle and select some --
    # np.random.shuffle(splits)
    splits = splits[:3]
    splits = np.array([[1,3],[0,4]])
    return splits

def create_nh_grids(BI,NH):
    nh_grid = np.arange(NH**2)
    nh_grid_rep = [nh_grid for _ in range(BI)]
    grids = np.meshgrid(*nh_grid_rep,copy=False)
    grids = np.array([grid.ravel() for grid in grids]).T
    grids = torch.LongTensor(grids)
    return grids

def create_grid_from_ndarrays(ndarrays):
    grids = np.meshgrid(*ndarrays)
    grids = np.array([grid.ravel() for grid in grids]).T
    grids = torch.LongTensor(grids)
    return grids

def create_powerset_pair_grids(S):
    indices = np.arange(S)
    subset_lg = chain.from_iterable(combinations(list(indices), r+1 ) for r in range(S-2,S))
    subset_lg = np.array([np.array(elem) for elem in list(subset_lg)])

    subset_sm = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(0,2))
    subset_sm = np.array([np.array(elem) for elem in list(subset_sm)])

    l_indices = [np.array([i]) for i in indices]
    subset_ex = list(subset_lg) + list(l_indices)
    subset_ex.extend(l_indices)

    grids = np.array(np.meshgrid(*[subset_ex,subset_ex]))
    grids = np.array([grid.ravel() for grid in grids]).T
    return grids

def create_n_grids(BI):
    # -- adjust for ref patch --
    indices = np.arange(BI+1)#np.r_[[0],burst_indices+1]
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
    l_indices.append(list(np.arange(BI+1)))

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


def get_ref_nh(NH): return NH**2//2 + NH//2*(NH%2==0)

def insert_nh_middle(best_indices_b,NH,BI):
    mid_NH = torch.LongTensor([get_ref_nh(NH)])
    left,right = np.r_[:BI//2],np.r_[BI//2:BI]
    return torch.cat([best_indices_b[[left]],mid_NH,best_indices_b[[right]]],
                     dim=0).type(torch.long)    

def insert_n_middle( burst_indices, N ):
    ngrid = np.arange(N)

    args_left = np.where(N//2 < burst_indices)[0]
    args_right = np.where(N//2 > burst_indices)[0]

    left = torch.LongTensor(burst_indices[args_left])
    right = torch.LongTensor(burst_indices[args_right])
    mid_N = torch.LongTensor([N//2])

    return torch.cat([left,mid_N,right],dim=0).type(torch.long)    

