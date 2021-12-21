

# -- python imports --
import torch
import numpy as np
from einops import rearrange

# -- project imports --
from align.nnf import compute_nnf

# -- local imports --
from .read_paths import read_nnf_paths
from .data_io import read_nnf,write_nnf
from .checks import check_nnf_data

def load_nnf_burst(frame_ids,vpaths,lpaths):

    # -- loop over frames --
    nframes = len(burst)
    assert nframes == len(frame_ids),"equal frame ids for burst and nnf"
    ref_fid = frame_ids[nframes//2]
    nnf_vals,nnf_locs = [],[]
    for t in range(nframes):

        # -- load image and compute nnf --
        fid = '%02d' % frame_ids[t]

        # -- get that nnf --
        _nnf_vals,_nnf_locs = read_nnf(vpaths[t],lpaths[t])

        # -- append to frame burst sample --
        nnf_vals.append(_nnf_vals)
        nnf_locs.append(_nnf_locs)

    nnf_vals = np.stack(nnf_vals)
    nnf_locs = np.stack(nnf_locs)

    return nnf_vals,nnf_locs

def create_nnf_burst(burst,burst_id,frame_ids,path_nnf,nnf_ps,nnf_K,icrop,isize):

    # -- loop over frames --
    nframes = len(burst)
    assert nframes == len(frame_ids),"equal frame ids for burst and nnf"
    ref_fid = "%02d" % frame_ids[nframes//2]
    ref_frame = burst[nframes//2]
    nnf_vals,nnf_locs = [],[]
    for t in range(nframes):

        # -- load image and compute nnf --
        fid = '%02d' % frame_ids[t]
        img = burst[t]

        # -- read paths --
        if not path_nnf.exists(): path_nnf.mkdir(parents=True)
        vpaths,lpaths = read_nnf_paths(burst_id,ref_fid,fid,path_nnf,nnf_K,icrop)

        # -- get that nnf --
        _nnf_vals,_nnf_locs = create_nnf_from_paths(ref_frame,img,nnf_ps,nnf_K,burst_id,
                                                    isize,ref_fid,fid,path_nnf,
                                                    vpaths,lpaths)

        # -- append to frame burst sample --
        nnf_vals.append(_nnf_vals)
        nnf_locs.append(_nnf_locs)

    nnf_vals = np.stack(nnf_vals)
    nnf_locs = np.stack(nnf_locs)

    return nnf_vals,nnf_locs


def create_nnf_from_paths(ref_frame,img,nnf_ps,nnf_K,burst_id,
                          isize,ref_fid,fid,path_nnf,vpaths,lpaths):

    # -- read into memory --
    nnf_vals,nnf_locs = read_nnf(vpaths,lpaths)

    # "ishape" might need to be "icrop"

    # -- check nnfs --
    ishape = ref_frame.shape
    if isize is None: isize = ishape[:2]
    looks_good = check_nnf_data(nnf_vals,nnf_locs,isize)
    if not (looks_good):
        print(f"ID: {burst_id}.{fid} is not looking good.")

        # -- format images --
        ref_frame = torch.FloatTensor(ref_frame)
        img = torch.FloatTensor(img)
        ref_frame = rearrange(ref_frame,'h w c -> c h w')
        img = rearrange(img,'h w c -> c h w')
        assert img.shape[0] in [1,3], "first channel must be the color."

        # -- run batch --
        nnf_vals,nnf_locs = compute_nnf(ref_frame,img,nnf_ps,nnf_K)

        # -- reshape (no image or frame dims) --
        nnf_vals = rearrange(nnf_vals,'1 1 h w k -> k h w')
        nnf_locs = rearrange(nnf_locs,'1 1 h w k two -> k h w two')
        write_nnf(nnf_vals,nnf_locs,vpaths,lpaths)

        print(f"ID: {burst_id}.{fid} is fixed!")

    # -- reshape --
    nnf_vals = rearrange(nnf_vals,'k h w -> k h w')
    nnf_locs = rearrange(nnf_locs,'k h w two -> k h w two')

    return nnf_vals,nnf_locs


