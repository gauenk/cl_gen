"""

We may recompute an NNF when the file exists,
causing a PyTorch error.

Just rerun the "write...nnf" function
to remove the error.

"""

# -- pyton imports --
import numpy as np
from einops import rearrange

# -- pytorch imports --
import torch

# -- project imports --
from align.nnf import compute_nnf
from .utils import read_ishape

def check_valid_burst_nnf(burst_id,fstart,nframes,paths,nnf_K,check_data=True):
    """
    Ensure all nnf fields are valid.

    set "check_data" to False for a faster validation.
    """
    ref_fid = '%02d' % int(fstart+nframes//2)
    frame_ids = np.arange(fstart,fstart+nframes)
    for t in range(nframes):
        fid = '%02d' % frame_ids[t]
        if check_nnf_exists(burst_id,ref_fid,fid,paths.nnf,nnf_K) is False:
            return False
        if check_data:
            nnfs = read_nnf(burst_id,ref_fid,fid,paths.nnf,nnf_K)
            ishape = read_ishape(paths.images,burst_id,fid)
            looks_good = check_nnf_data(nnfs,ishape)
            if looks_good is False:
                return False
    return True

def check_nnf_data(nnfs,ishape):
    recomp_nnf = False
    if nnfs is None:
        recomp_nnf = True
    elif nnfs[1].ndim != 4:
        recomp_nnf = True        
    else:
        k,nnf_h,nnf_w,two = nnfs[1].shape
        img_h,img_w,c = ishape
        eq_h = nnf_h == img_h
        eq_w = nnf_w == img_w
        if not(eq_h) or not(eq_w):
            recomp_nnf = True
    looking_good = not(recomp_nnf)
    return looking_good

def check_nnf_exists(burst_id,ref_fid,fid,path_nnf,K):
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
        vals_k,locs_k = vals[k],locs[k]
        torch.save(vals_k,path_nnf_val)
        torch.save(locs_k,path_nnf_loc)

def get_nnf(ref_frame,img,burst_id,ref_fid,fid,path_nnf,nnf_ps,nnf_K):
    """
    We want the NNF.

    (i) read or (ii) compute and cachce the nnf
    """
    if not path_nnf.exists(): path_nnf.mkdir(parents=True)

    # -- get into memory --
    nnfs = read_nnf(burst_id,ref_fid,fid,path_nnf,nnf_K)
    ishape = ref_frame.shape
    looks_good = check_nnf_data(nnfs,ishape)

    if not (looks_good):
        print(f"ID: {burst_id}.{fid} is not looking good.")

        # -- format images --
        ref_frame = torch.FloatTensor(ref_frame)
        img = torch.FloatTensor(img)
        ref_frame = rearrange(ref_frame,'h w c -> c h w')
        img = rearrange(img,'h w c -> c h w')

        # -- run batch --
        nnf_vals,nnf_locs = compute_nnf(ref_frame,img,nnf_ps,nnf_K)

        # -- reshape (no image or frame dims) --
        nnf_vals = rearrange(nnf_vals,'1 1 h w k -> k h w')
        nnf_locs = rearrange(nnf_locs,'1 1 h w k two -> k h w two')
        write_nnf(nnf_vals,nnf_locs,burst_id,ref_fid,fid,path_nnf,nnf_K)

    else:
        nnf_vals,nnf_locs = nnfs

    # -- reshape --
    nnf_vals = rearrange(nnf_vals,'k h w -> k h w')
    nnf_locs = rearrange(nnf_locs,'k h w two -> k h w two')

    return nnf_vals,nnf_locs


