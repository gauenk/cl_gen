
# -- python imports --
import torch
import numpy as np


def read_nnf_burst(lpaths,vpaths):
    locs,vals = [],[]
    nframes = len(lpaths)
    for t in range(nframes):
        _lpaths = lpaths[t]
        _vpaths = None if vpaths is None else vpaths[t]
        _locs,_vals = read_nnf(_lpaths,_vpaths)
        locs.append(_locs)
        vals.append(_vals)
    locs = np.stack(locs)
    vals = np.stack(vals)
    return locs,vals

def read_nnf(lpaths,vpaths):

    # -- set K --
    K = len(lpaths)

    # -- load locs --
    locs = []
    for k in range(K):
        path_nnf_loc = lpaths[k]
        if not path_nnf_loc.exists(): return None
        locs_k = torch.load(path_nnf_loc)
        locs.append(locs_k)
    locs = np.stack(locs,axis=0)
    
    # -- load vals --
    if vpaths is None:
        k,h,w,two = locs.shape
        vals = np.zeros((k,h,w))
    else:
        vals = []
        for k in range(K):
            path_nnf_val = vpaths[k]
            if not path_nnf_val.exists(): return None
            vals_k = torch.load(path_nnf_val)
            vals.append(vals_k)
        vals = np.stack(vals,axis=0)

    return locs,vals

def write_nnf(vals,locs,lpaths,vpaths):

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
