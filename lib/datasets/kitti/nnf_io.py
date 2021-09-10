

# -- pyton imports --
import numpy as np

# -- pytorch imports --
import torch

# -- project imports --
from align.nnf import compute_nnf

def check_nnf(burst_id,ref_fid,fid,path_nnf,K):
    for k in range(K):
        loc_str = "loc_%s_%s_%s_%02d.pt" % (burst_id,ref_fid,fid,k)
        val_str = "val_%s_%s_%s_%02d.pt" % (burst_id,ref_fid,fid,k)
        path_nnf_loc = path_nnf / loc_str
        path_nnf_val = path_nnf / val_str
        if not path_nnf_loc.exists() or not path_nnf_val.exists():
            return False
    return True

def read_nnf(burst_id,ref_fid,fid,path_nnf,K):

    # -- load nnf --
    vals,locs = [],[]
    for k in range(K):
        loc_str = "loc_%s_%s_%s_%02d.pt" % (burst_id,ref_fid,fid,k)
        val_str = "val_%s_%s_%s_%02d.pt" % (burst_id,ref_fid,fid,k)
        path_nnf_loc = path_nnf / loc_str
        path_nnf_val = path_nnf / val_str
        if not path_nnf_loc.exists() or not path_nnf_val.exists():
            return None
        vals_k = torch.load(path_nnf_val)
        locs_k = torch.load(path_nnf_loc)
        vals.append(vals_k)
        locs.append(locs_k)
    vals = np.stack(vals,axis=0)
    locs = np.stack(locs,axis=0)
    return vals,locs

def write_nnf(vals,locs,burst_id,ref_fid,fid,path_nnf,K):

    # -- path for nnf --
    for k in range(K):
        val_str = "val_%s_%s_%s_%02d.pt" % (burst_id,ref_fid,fid,k)
        loc_str = "loc_%s_%s_%s_%02d.pt" % (burst_id,ref_fid,fid,k)
        path_nnf_val = path_nnf / val_str
        path_nnf_loc = path_nnf / loc_str
        vals_k,locs_k = vals[:,:,k],locs[:,:,k]
        torch.save(vals_k,path_nnf_val)
        torch.save(locs_k,path_nnf_loc)

def get_nnf(ref_frame,img,burst_id,ref_fid,fid,path_nnf,nnf_ps,nnf_K):
    """
    We want the NNF.

    (i) read or (ii) compute and cachce the nnf
    """
    if not path_nnf.exists(): path_nnf.mkdir(parents=True)

    nnfs = read_nnf(burst_id,ref_fid,fid,path_nnf,nnf_K)
    if nnfs is None:
        ref_frame = torch.FloatTensor(ref_frame)
        img = torch.FloatTensor(img)
        nnf_vals,nnf_locs = compute_nnf(ref_frame,img,nnf_ps)
        write_nnf(nnf_vals,nnf_locs,burst_id,ref_fid,fid,path_nnf,nnf_K)
    else:
        nnf_vals,nnf_locs = nnfs
    return nnf_vals,nnf_locs


