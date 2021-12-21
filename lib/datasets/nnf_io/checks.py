
# -- python imports --
import numpy as np

# -- local imports --
from .read_paths import read_nnf_paths
from .data_io import read_nnf
from .utils import read_ishape


def check_valid_burst_nnf(burst_id,frame_ids,path_nnf,nnf_K,
                          isize,icrop,fpaths,check_data=True):
    """
    Ensure all nnf fields are valid.

    set "check_data" to False for a faster validation.
    """
    nframes = len(frame_ids)
    ref_fid = '%02d' % frame_ids[nframes//2]
    for t in range(nframes):
        fid = '%02d' % frame_ids[t]
        if check_nnf_exists(burst_id,ref_fid,fid,path_nnf,nnf_K,icrop) is False:
            return False
        if check_data:
            vpaths,lpaths = read_nnf_paths(burst_id,ref_fid,fid,path_nnf,nnf_K,icrop)
            vals,locs = read_nnf(vpaths,lpaths)
            ishape = read_ishape(fpaths[t])
            if isize is None: isize = ishape[:2]
            looks_good = check_nnf_data(vals,locs,isize)
            if looks_good is False:
                return False
    return True

def check_nnf_data(vals,locs,ishape):
    recomp_nnf = False
    if vals is None or locs is None:
        recomp_nnf = True
    elif locs.ndim != 4:
        recomp_nnf = True
    else:
        k,nnf_h,nnf_w,two = locs.shape
        img_h,img_w = ishape
        eq_h = nnf_h == img_h
        eq_w = nnf_w == img_w
        if not(eq_h) or not(eq_w):
            recomp_nnf = True
    looking_good = not(recomp_nnf)
    return looking_good


def check_nnf_exists(burst_id,ref_fid,fid,path_nnf,K,icrop):
    vpaths,lpaths = read_nnf_paths(burst_id,ref_fid,fid,path_nnf,K,icrop)
    for k in range(K):
        path_nnf_val = vpaths[k]
        path_nnf_loc = lpaths[k]
        if not path_nnf_val.exists() or not path_nnf_loc.exists() :
            return False
    return True
