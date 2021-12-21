
# -- python imports --
import nvtx
import numpy as np
from einops import rearrange,repeat

# -- pytorch imports --
import torch

# -- faiss imports --
import faiss

# -- project imports --
from pyutils import tile_patches

def compute_burst_nnf(burst,ref_t,patchsize,K=10,gpuid=0,pxform=None):
    T,B,C,H,W = burst.shape
    npix = H*W
    ref_img = burst[ref_t]
    from_image = ref_img
    vals,locs = [],[]
    for t in range(T):
        if t == ref_t:
            vals_t = torch.zeros((1,B,H,W,K))
            indices = np.c_[np.unravel_index(np.arange(npix),(H,W))]
            # (AT END) swap: (rows,cols) -> (cols,rows) aka (y,x) -> (x,y)
            indices = indices.reshape(H,W,2)
            locs_t = repeat(indices,'h w two -> 1 b h w k two',b=B,k=K)
            locs_t[...,:] = locs_t[...,::-1] # row,cols -> cols,rows
        else:
            to_image = burst[t]
            vals_t,locs_t = compute_batch_nnf(from_image,to_image,
                                              patchsize,K,gpuid,pxform)
        vals.append(vals_t)
        locs.append(locs_t)
    vals = np.concatenate(vals,axis=0)
    locs = np.concatenate(locs,axis=0)
    # print("burst_nnf.shape", locs.shape)
    return vals,locs

def compute_batch_nnf(ref_img,prop_img,patchsize,K=10,gpuid=0,pxform=None):
    B = ref_img.shape[0]
    locs,vals = [],[]
    for b in range(B):
        ref_img_b = ref_img[b]
        prop_img_b = prop_img[b]
        vals_b,locs_b = compute_nnf(ref_img_b,prop_img_b,
                                    patchsize,K,gpuid,pxform)
        vals.append(vals_b)
        locs.append(locs_b)
    vals = np.concatenate(vals,axis=1)
    locs = np.concatenate(locs,axis=1)
    return vals,locs

@nvtx.annotate("compute_nnf", color="green")
def compute_nnf(ref_img,prop_img,patchsize,K=10,gpuid=0,pxform=None):
    """
    Compute the Nearest Neighbor Field for Optical Flow
    """
    C,H,W = ref_img.shape
    B,T = 1,1

    # -- tile patches --
    query = repeat(ref_img,'c h w -> 1 1 c h w')
    q_patches = tile_patches(query,patchsize,pxform).pix.cpu().numpy()
    B,N,R,ND = q_patches.shape
    query = rearrange(q_patches,'b t r nd -> (b t r) nd')

    db = repeat(prop_img,'c h w -> 1 1 c h w')
    db_patches = tile_patches(db,patchsize,pxform).pix.cpu().numpy()
    Bd,Nd,Rd,NDd = db_patches.shape
    database = rearrange(db_patches,'b t r nd -> (b t r) nd')

    # -- faiss setup --
    res = faiss.StandardGpuResources()
    faiss_cfg = faiss.GpuIndexFlatConfig()
    faiss_cfg.useFloat16 = False
    faiss_cfg.device = gpuid
    faiss.cvar.distance_compute_blas_threshold = 40

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
            # (AT END) swap: (rows,cols) -> (cols,rows) aka (y,x) -> (x,y)
            locs_bt = rearrange(locs_bt,'(h w) k two -> h w k two',h=H)

            vals.append(vals_bt)
            locs.append(locs_bt)

    vals = rearrange(vals,'(b t) h w k -> b t h w k',b=B)
    locs = rearrange(locs,'(b t) h w k two -> b t h w k two',b=B)
    locs[...,:] = locs[...,::-1] # (HERE) row,cols -> cols,rows

    return vals,locs
