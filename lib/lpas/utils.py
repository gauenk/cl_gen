
import numpy as np
import pandas as pd

import torch

def get_ref_block_index(nblocks): return nblocks**2//2 + (nblocks//2)*(nblocks%2==0)

def get_block_arangements(nblocks,nframes):
    # -- create grid over neighboring blocks --
    nh_grid = np.arange(nblocks**2)
    nh_grid_rep = [nh_grid for _ in range(nframes-1)]
    grids = np.meshgrid(*nh_grid_rep,copy=False)
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- fix index for reference patch --
    REF_N = get_ref_block_index(nblocks)    
    ref_index = REF_N * torch.ones(grids.shape[0])
    M = ( grids.shape[1] ) // 2
    grids = np.c_[grids[:,:M],ref_index,grids[:,M:]]

    # -- convert for torch --
    grids = torch.LongTensor(grids)
    return grids

def create_meshgrid(lists):
    # -- num lists --
    L = len(lists)

    # -- tokenize each list --
    codes,uniques = [],[]
    for l in lists:
        l_codes,l_uniques = pd.factorize(l)
        codes.append(l_codes)
        uniques.append(l_uniques)

    # -- meshgrid and flatten --
    lmesh = np.meshgrid(*codes)
    int_mesh = [grid.ravel() for grid in lmesh]

    # -- convert back to tokens --
    mesh = [uniques[i][int_mesh[i]] for i in range(L)]

    # -- "transpose" the axis to iter goes across original lists --
    mesh_T = []
    L,M = len(mesh),len(mesh[0])
    for m in range(M):
        mesh_m = []
        for l in range(L):
            mesh_m.append(mesh[l][m])
        mesh_T.append(mesh_m)

    return mesh_T
