
# -- python imports --
import numpy as np

# -- local imports --
from .read_paths import read_nnf_paths
from .data_io import read_nnf
from .utils import read_ishape


def check_valid_burst_nnf(burst_id,fstart,nframes,path_nnf,nnf_K,fpaths,check_data=True):
    """
    Ensure all nnf fields are valid.

    set "check_data" to False for a faster validation.
    """
    ref_fid = '%02d' % int(fstart+nframes//2)
    frame_ids = np.arange(fstart,fstart+nframes)
    for t in range(nframes):
        fid = '%02d' % frame_ids[t]
        if check_nnf_exists(burst_id,ref_fid,fid,path_nnf,nnf_K) is False:
            return False
        if check_data:
            lpaths,vpaths = read_nnf_paths(burst_id,ref_fid,fid,path_nnf,nnf_K)
            nnfs = read_nnf(lpaths,vpaths)
            ishape = read_ishape(fpaths[t])
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
        print("nnfs[1].shape: ",nnfs[1].shape)
        eq_h = nnf_h == img_h
        eq_w = nnf_w == img_w
        if not(eq_h) or not(eq_w):
            recomp_nnf = True
    looking_good = not(recomp_nnf)
    return looking_good


def check_nnf_exists(burst_id,ref_fid,fid,path_nnf,K):
    lpaths,vpaths = read_nnf_paths(burst_id,ref_fid,fid,path_nnf,K)
    for k in range(K):
        path_nnf_loc = lpaths[k]
        path_nnf_val = vpaths[k]
        if not path_nnf_loc.exists() or not path_nnf_val.exists():
            return False        
    return True
