
# -- python imports --
import random
import numpy as np
import numpy.random as npr
import pandas as pd
from einops import rearrange,repeat

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils.sobel import apply_sobel_filter

def print_tensor_stats(name,tensor):
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    t_mean = tensor.mean().item()
    print(name,t_min,t_max,t_mean)

def sample_good_init_tl(full_image,P,patchsize,nblocks):
    # -- shape to include batch --
    if len(full_image.shape) == 3:
        full_image = full_image.unsqueeze(0)
        
    # -- get edges --        
    B,C,H,W = full_image.shape
    edges = apply_sobel_filter(full_image)
    B,H,W = edges.shape
    S = 3*P

    # -- sample based on edges and patchsize --
    buf,eps = 3,1e-13
    fs = patchsize//2 + buf + nblocks//2
    inner = edges[:,fs:-fs,fs:-fs]
    topP = torch.topk(inner.reshape(B,-1),S,largest=True).values
    init_tl_list = []
    for b in range(B):
        init_tl_b = []
        for s in range(S):
            maxs = topP[b,s]
            indices = torch.nonzero( torch.abs(inner[b] - maxs) < eps)
            for i in range(indices.shape[0]):
                init_tl = [int(x)+fs-patchsize//2 for x in indices[i]]
                init_tl_b.append(init_tl)
        order = npr.permutation(len(init_tl_b))[:P]
        init_tl_b = [init_tl_b[x] for x in order]
        init_tl_list.append(init_tl_b)
    init_tl_list = torch.LongTensor(init_tl_list)
    return init_tl_list

def get_sobel_patches(burst,nblocks,R,patchsize):
    ps,(nframes,B) = patchsize,burst.shape[:2]
    image = torch.mean(burst,dim=0)
    indices = sample_good_init_tl(image,R,ps,nblocks)
    patches = []
    for b in range(B):
        indices_b = indices[b]
        patches_b = crop_burst_to_blocks(burst[:,b],nblocks,indices_b,ps)
        patches.append(patches_b)
    patches = torch.stack(patches,dim=0)
    patches = rearrange(patches,'b t h r c p1 p2 -> b r t h c p1 p2')
    return patches,indices
    
def crop_burst_to_blocks(full_burst,nblocks,init_tl_list,patchsize):
    P,H,ps = len(init_tl_list),nblocks,patchsize
    blocks = []
    for p in range(P):
        t,l = init_tl_list[p]
        blocks_p = []
        for dy in range(-H//2+1,H//2+1):
            for dx in range(-H//2+1,H//2+1):
                crop = tvF.crop(full_burst,t+dy,l+dx,ps,ps)
                # if dy == 0 and dx == 0: save_image(crop,f"crop_{p}.png")
                blocks_p.append(crop)
        blocks_p = torch.stack(blocks_p,dim=0)
        blocks.append(blocks_p)
    blocks = torch.stack(blocks,dim=0)
    blocks = rearrange(blocks,'p h2 t c h w -> t h2 p c h w',p=P)
    return blocks

def random_sample_sim_search_block_search_space(nframes,nblocks):
    pass

def save_image(images,fn,normalize=True,vrange=None):
    if len(images.shape) > 4:
        C,H,W = images.shape[-3:]
        images = images.reshape(-1,C,H,W)
    if vrange is None:
        tv_utils.save_image(images,fn,normalize=normalize)
    else:
        tv_utils.save_image(images,fn,normalize=normalize,range=vrange)

def get_ref_block_index(nblocks): return nblocks**2//2 + (nblocks//2)*(nblocks%2==0)

def get_small_test_block_arangements(bss_dir,nblocks,nframes,tcount,size,difficult=False):
    if not bss_dir.exists(): bss_dir.mkdir()
    bss_fn = bss_dir / f"block_arange_{nblocks}b_{nframes}f_{tcount}t_{size}s.npy"
    REF_H = get_ref_block_index(nblocks)
    if bss_fn.exists():
        print(f"[Reading bss]: {bss_fn}")
        block_search_space = np.load(bss_fn,allow_pickle=True)
        bss = block_search_space
    else:
        block_search_space = get_block_arangements_subset(nblocks,nframes,tcount,difficult=difficult)
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

def get_block_arangements_freeze(nframes,nblocks,fix_frames):
    # -- create grid over neighboring blocks --
    bl_grid = np.arange(nblocks**2)
    bl_grid_rep = []
    for t in range(nframes):
        if t in fix_frames.idx:
            list_idx = fix_frames.idx.index(t)
            bl_grid_rep.append([fix_frames.vals[list_idx]])
        else:
            rand = np.random.permutation(len(bl_grid))
            bl_grid_rep.append(rand)    
    grids = np.meshgrid(*bl_grid_rep,copy=False)
    grids = np.array([grid.ravel() for grid in grids]).T
    grids = torch.LongTensor(grids)
    return grids

def get_block_arangements_split(nframes,nblocks,fix_frames):
    pass

def get_block_arangements_subset(nblocks,nframes,tcount=10,difficult=False):
    # -- create grid over neighboring blocks --
    nh_grid = np.arange(nblocks**2)
    REF_N = get_ref_block_index(nblocks)    

    # -- rand for nframes-1 --
    nh_grid_rep = []
    for t in range(nframes-1):
        diff_valid = tcount < nblocks**2
        if difficult and diff_valid:
            mid_point = tcount//2
            rand = []
            for t_index in range(tcount):
                if t_index >= mid_point:
                    rand.append(REF_N-mid_point+t_index+1)
                else:
                    rand.append(REF_N-mid_point+t_index)
            rand = np.array(rand)
        else:
            rand = np.random.permutation(len(nh_grid))[:tcount]
        nh_grid_rep.append(rand)
    grids = np.meshgrid(*nh_grid_rep,copy=False)
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- fix index for reference patch --
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


