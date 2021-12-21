
# -- python imports --
from easydict import EasyDict as edict


def get_crop_str(icrop):
    if isinstance(icrop[0],str):
        crop_str = "%s_%s" % (icrop[0],icrop[1])
    else:
        crop_str = "%03d_%03d" % (icrop[0],icrop[1])
    return crop_str

def read_nnf_paths(burst_id,ref_fid,fid,path_nnf,K,icrop):
    vpaths,lpaths = [],[]
    crop_str = get_crop_str(icrop)
    for k in range(K):
        # -- use MUST have "ref_fid" be the [-3] elem back --
        val_str = "val_%s_%s_%s_%s_%02d.pt" % (burst_id,crop_str,ref_fid,fid,k)
        loc_str = "loc_%s_%s_%s_%s_%02d.pt" % (burst_id,crop_str,ref_fid,fid,k)
        vpaths.append(path_nnf/val_str)
        lpaths.append(path_nnf/loc_str)
    return vpaths,lpaths

def read_nnf_burst_paths(burst_id,frame_ids,path_nnf,K,icrop):

    # -- loop over frames --
    paths = edict()
    vpaths,lpaths = [],[]
    nframes = len(frame_ids)
    ref_fid = "%02d" % frame_ids[nframes//2]
    for t in range(nframes):

        # -- get nnfs --
        fid = '%02d' % frame_ids[t]
        _vpaths,_lpaths = read_nnf_paths(burst_id,ref_fid,fid,path_nnf,K,icrop)

        # -- append to frame burst sample --
        vpaths.append(_vpaths)
        lpaths.append(_lpaths)

    return vpaths,lpaths
