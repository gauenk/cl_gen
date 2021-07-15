
# -- python imports --
import numpy as np
from einops import rearrange,repeat

# -- pytorch imports --
import torch

# -- faiss imports --
import faiss

# -- project imports --
from pyutils import tile_patches

def compute_burst_nnf(burst,ref_t,patchsize,K=10,gpuid=0):
    T,B,C,H,W = burst.shape
    ref_img = burst[ref_t]
    vals,locs = [],[]
    for t in range(T):
        if t == ref_t: continue
        img = burst[t]
        vals_t,locs_t = compute_batch_nnf(ref_img,img,patchsize,K,gpuid)
        vals.append(vals_t)
        locs.append(locs_t)
    vals = np.concatenate(vals,axis=0)
    locs = np.concatenate(locs,axis=0)
    return vals,locs

def compute_batch_nnf(ref_img,prop_img,patchsize,K=10,gpuid=0):
    B = ref_img.shape[0]
    locs,vals = [],[]
    for b in range(B):
        ref_img_b = ref_img[b]
        prop_img_b = prop_img[b]
        vals_b,locs_b = compute_nnf(ref_img_b,prop_img_b,patchsize,K,gpuid)
        vals.append(vals_b)
        locs.append(locs_b)
    vals = np.concatenate(vals,axis=1)
    locs = np.concatenate(locs,axis=1)
    return vals,locs

def compute_nnf(ref_img,prop_img,patchsize,K=10,gpuid=0):
    """
    Compute the Nearest Neighbor Field for Optical Flow
    """
    C,H,W = ref_img.shape
    B,T = 1,1

    # -- tile patches --
    query = repeat(prop_img,'c h w -> 1 1 c h w')
    q_patches = tile_patches(query,patchsize).pix.cpu().numpy()
    B,N,R,ND = q_patches.shape
    query = rearrange(q_patches,'b t r nd -> (b t r) nd')

    db = repeat(ref_img,'c h w -> 1 1 c h w')
    db_patches = tile_patches(db,patchsize).pix.cpu().numpy()
    Bd,Nd,Rd,NDd = db_patches.shape
    database = rearrange(db_patches,'b t r nd -> (b t r) nd')

    # -- faiss setup --
    res = faiss.StandardGpuResources()
    faiss_cfg = faiss.GpuIndexFlatConfig()
    faiss_cfg.useFloat16 = False
    faiss_cfg.device = gpuid

    # -- create database --
    gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
    gpu_index.add(database)
    
    # -- execute search --
    D,I = gpu_index.search(query,K)
    D = rearrange(D,'(b t r) k -> b t r k',b=B,t=T)
    I = rearrange(I,'(b t r) k -> b t r k',b=B,t=T)

    # -- get nnf (x,y) from I --
    vals,locs = [],[]
    for b in range(B):
        for t in range(T):

            D_bt,I_bt = D[b][t],I[b][t]

            vals_bt = rearrange(D_bt,'(h w) k -> h w k',h=H)

            locs_bt = np.unravel_index(I_bt,(H,W)) # only works with B,T == 1
            locs_bt = np.stack(locs_bt,axis=-1)
            locs_bt = rearrange(locs_bt,'(h w) k two -> h w k two',h=H)

            vals.append(vals_bt)
            locs.append(locs_bt)

    vals = rearrange(vals,'(b t) h w k -> b t h w k',b=B)
    locs = rearrange(locs,'(b t) h w k two -> b t h w k two',b=B)

    return vals,locs
