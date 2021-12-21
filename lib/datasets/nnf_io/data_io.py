
# -- python imports --
import torch
import shutil
import numpy as np
from pathlib import Path


def read_nnf_burst(vpaths,lpaths):
    vals,locs = [],[]
    nframes = len(lpaths)
    for t in range(nframes):
        _vpaths = None if vpaths is None else vpaths[t]
        _lpaths = lpaths[t]
        _vals,_locs = read_nnf(_vpaths,_lpaths)
        vals.append(_vals)
        locs.append(_locs)
    vals = np.stack(vals)
    locs = np.stack(locs)
    return vals,locs

def try_int_d_fmt(path_nnf):
    path_nnf = str(path_nnf)
    srid = path_nnf.split("_")[-3]
    rid = int(srid)
    if rid < 10 and len(srid) == 2:
        stem = Path(path_nnf).stem
        rstem = stem.replace("%02d"%rid,"%d"%rid,1)
        path_nnf = path_nnf.replace(stem,rstem)
    return Path(path_nnf)

def read_nnf(vpaths,lpaths):

    # -- set K --
    K = len(lpaths)

    # -- load locs --
    locs = []
    for k in range(K):
        path_nnf_loc = lpaths[k]
        if not path_nnf_loc.exists():
            path_nnf_loc_fmt = try_int_d_fmt(path_nnf_loc) # allow for %d v.s. %02d
            if not(path_nnf_loc_fmt.exists()): return None,None
            else: move_file(path_nnf_loc_fmt,path_nnf_loc)
        locs_k = torch.load(path_nnf_loc)
        locs.append(locs_k)
    locs = np.stack(locs,axis=0)

    # -- load vals --
    if vpaths is None:
        k,h,w,two = locs.shape # this is why locs 1st
        vals = np.zeros((k,h,w))
    else:
        vals = []
        for k in range(K):
            path_nnf_val = vpaths[k]
            if not path_nnf_val.exists():
                path_nnf_val_fmt = try_int_d_fmt(path_nnf_val) # allow for %d v.s. %02d
                if not(path_nnf_val.exists()): return None,None
                else: copy_file(path_nnf_val_fmt,path_nnf_val)
            vals_k = torch.load(path_nnf_val)
            vals.append(vals_k)
        vals = np.stack(vals,axis=0)

    return vals,locs

def move_file(from_file,to_file):
    print("from file: ",from_file)
    print("to file: ",to_file)
    shutil.move(from_file,to_file)

def write_nnf(vals,locs,vpaths,lpaths):

    # -- check sizes --
    assert len(vals) == len(locs),"checking K"
    assert len(lpaths) == len(locs),"checking K"
    assert len(vpaths) == len(vals),"checking K"
    assert len(lpaths) == len(vpaths),"checking K"

    # -- path for nnf --
    K = len(vals)
    for k in range(K):
        path_nnf_val = vpaths[k]
        path_nnf_loc = lpaths[k]
        vals_k,locs_k = vals[k],locs[k]
        torch.save(vals_k,path_nnf_val)
        torch.save(locs_k,path_nnf_loc)
