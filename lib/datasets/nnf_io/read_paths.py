
# -- python imports --
from easydict import EasyDict as edict


def read_nnf_paths(burst_id,ref_fid,fid,path_nnf,K):
    lpaths,vpaths = [],[]
    for k in range(K):
        loc_str = "loc_%s_%s_%s_%02d.pt" % (burst_id,ref_fid,fid,k)
        val_str = "val_%s_%s_%s_%02d.pt" % (burst_id,ref_fid,fid,k)
        lpaths.append(path_nnf/loc_str)
        vpaths.append(path_nnf/val_str)
    return lpaths,vpaths

def read_nnf_burst_paths(burst_id,frame_ids,path_nnf,K):

    # -- loop over frames --            
    paths = edict()
    lpaths,vpaths = [],[]
    nframes = len(frame_ids)
    ref_fid = frame_ids[nframes//2]
    for t in range(nframes):

        # -- get nnfs --
        fid = '%02d' % frame_ids[t]
        _lpaths,_vpaths = read_nnf_paths(burst_id,ref_fid,fid,path_nnf,K)

        # -- append to frame burst sample --
        lpaths.append(_lpaths)
        vpaths.append(_vpaths)

    return lpaths,vpaths
