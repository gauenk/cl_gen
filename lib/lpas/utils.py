
import random
import numpy as np
import pandas as pd


# -- pytorch imports --
import torch
import torchvision.utils as tv_utils

def get_patches(burst):
    pass
    
def random_sample_sim_search_block_search_space(nframes,nblocks):
    pass

def save_image(images,fn,normalize=True,vrange=None):
    if len(images.shape) > 4:
        C,H,W = images.shape[-3:]
        images = images.view(-1,C,H,W)
    if vrange is None:
        tv_utils.save_image(images,fn,normalize=normalize)
    else:
        tv_utils.save_image(images,fn,normalize=normalize,range=vrange)

def get_ref_block_index(nblocks): return nblocks**2//2 + (nblocks//2)*(nblocks%2==0)

def get_small_test_block_arangements(bss_dir,nblocks,nframes,tcount,size):
    bss_fn = bss_dir / "block_arange_{nblocks}b_{nframes}f_{tcount}t_{size}s.npy"
    REF_H = get_ref_block_index(nblocks)
    block_search_space = get_block_arangements_subset(nblocks,nframes,tcount)
    if bss_fn.exists():
        print(f"Reading bss {bss_fn}")
        block_search_space = np.load(bss_fn,allow_pickle=True)
        bss = block_search_space
    else:
        bss = block_search_space
        print(f"Original block search space: [{len(bss)}]")
        if len(block_search_space) >= size:
            rand_blocks = random.sample(list(block_search_space),size)
            block_search_space = [np.array([REF_H]*nframes),] # include gt
            block_search_space.extend(rand_blocks)
        bss = block_search_space
        print(f"Writing block search space: [{bss_fn}]")
        np.save(bss_fn,np.array(block_search_space))
    print(f"Search Space Size: {len(block_search_space)}")
    return bss

def get_block_arangements_freeze(nblocks,fix_frames):
    # -- create grid over neighboring blocks --
    bl_grid = np.arange(nblocks**2)

    nframes = len(srch_frames) + len(fix_frames)
    bl_grid_rep = []
    for t in range(nframes-1):
        if t in fix_frames.keys():
            bl_grid_rep.append(fix_frames_val[t])
        else:
            rand = np.random.permutation(len(bl_grid))
            bl_grid_rep.append(rand)    
    grids = np.meshgrid(*nh_grid_rep,copy=False)
    grids = np.array([grid.ravel() for grid in grids]).T
    return bl_grid_rep

def get_block_arangements_subset(nblocks,nframes,tcount=10):
    # -- create grid over neighboring blocks --
    nh_grid = np.arange(nblocks**2)

    # -- rand for nframes-1 --
    nh_grid_rep = []
    for t in range(nframes-1):
        rand = np.random.permutation(len(nh_grid))[:tcount]
        nh_grid_rep.append(rand)
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

def rand_sample_two(R):
    if T == 2:
        order = npr.permutation(T)
        input_idx = order[0]
        i,j = order[1],order[1]
    else:
        input_idx = T//2
        i,j = random.sample(picks,2)
    # i,j = random.sample(list(range(burst.shape[0])),2)
    return input_idx

def pixel_shuffle_uniform(burst,B):
    T,C,H,W = burst.shape
    R = H*W
    indices = torch.randint(0,T,(B,R),device=burst.device)
    shuffle = torch.zeros((C,B,R),device=burst.device)
    along = torch.arange(T)
    cburst = rearrange(burst,'t c h w -> c t (h w)')
    for c in range(C):
        shuffle[c] = torch.gather(cburst[c],0,indices)
    shuffle = rearrange(shuffle,'c b (h w) -> b c h w',h=H)
    # save_image(burst,"burst.png")
    # save_image(shuffle,"shuffle.png")
    return shuffle

def pixel_shuffle_perm(burst):
    T,C,H,W = burst.shape
    R = H*W
    order = torch.stack([torch.randperm(T,device=burst.device) for _ in range(R)],dim=1)
    order = repeat(order,'b r -> b c r',c=C).long()
    cburst = rearrange(burst,'t c h w -> t c (h w)')
    target = torch.zeros_like(cburst,device=cburst.device)
    target.scatter_(0,order,cburst)
    target = rearrange(target,'t c (h w) -> t c h w',h=H)
    return target

