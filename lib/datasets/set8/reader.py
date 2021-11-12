
# -- python imports --
import glob
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils import get_img_coords
from datasets.nnf_io import create_nnf_burst,check_valid_burst_nnf,read_nnf_burst_paths
from datasets.nnf_io import read_nnf_burst,read_nnf


CHECK_NNF_DATA = True # make true for nnf checking; false for faster checking

def read_split_ids(sdir,split):

    # -- get filename --
    if split == "train":
        fn = "train.txt"
    elif split == "val":
        fn = "val.txt"
    elif split == "test" or split == "test-dev":
        fn = "test.txt"
    elif split == "all":
        fn = "all.txt"
        
    # -- get full path --
    fn = sdir / fn

    # -- read --
    dirs = []
    with open(fn,"r") as f:
        _dirs = f.readlines()
        for elem in _dirs:
            dirs.append(elem.strip())

    return dirs

def get_isize_str(isize):
    if not(isize is None):
        #print("Warning: we don't use crops right now for cbsd68.")
        isize_str = "%d_%d" % (isize[0],isize[1])
    else:
        isize_str = ""
    return isize_str

def get_param_str(isize,ps):
    isize_str = get_isize_str(isize)
    param_str = "%s_%d" % (isize_str,ps)
    return param_str

def read_burst(ipaths,isize):
    dyn_clean = []
    for ipath in ipaths:
        img = Image.open(ipath).convert("RGB")
        dyn_clean.append(tvF.to_tensor(img))
    dyn_clean = torch.stack(dyn_clean)
    if not(isize is None):
        dyn_clean = tvF.center_crop(dyn_clean,isize)
    return dyn_clean

def read_flow(fpaths,vpaths=None):

    # -- read all locs --
    locs,_ = read_nnf_burst(fpaths,vpaths)
    locs = rearrange(locs,'t k h w two -> two t k h w')
    ref_flows = locs2flow(locs)
    return ref_flows

def locs2flow(locs):
    two,t,k,h,w = locs.shape
    coords = get_img_coords(t,k,h,w).numpy()
    print("coords.shape: ",coords.shape)
    flows = locs - coords # [Delta row,Delta col]
    flows = np.flip(flows,axis=(-1,)) # [D row, D col]
    return flows

def read_files(idir,fdir,sdir,split,isize,ps,nframes):
    """
    Primary file called in the [dataset].py file

    """

    # -- read all files --
    groups = []
    idir_star = str(idir / "*")
    for fn in glob.glob(idir_star):
        group = Path(fn).stem
        groups.append(group)

    # -- read split ids --
    split_groups = read_split_ids(sdir,split)

    # -- init output var --
    paths = edict({'images':{},'flows':{},
                   'vals':{},'frame_ids':{}})

    # -- read image paths --
    out_nframes,all_eq = None,True
    for group in groups:
        if not(group in split_groups): continue
        ipaths,frame_ids = get_image_paths(idir/group,nframes)
        paths['images'][group] = ipaths
        paths['frame_ids'][group] = frame_ids
        out_nframes,all_eq = update_nframes(out_nframes,frame_ids,all_eq)

    # -- read flow paths --
    for group in groups:
        if not(group in split_groups): continue
        ipaths = paths['images'][group]
        frame_ids = paths['frame_ids'][group]
        fpaths,vpaths = get_flow_paths(fdir,group,isize,ps,ipaths,frame_ids)
        paths['flows'][group] = fpaths
        paths['vals'][group] = vpaths

    return paths,out_nframes,all_eq
    
def update_nframes(nframes,frame_ids,all_eq):
    if nframes is None:
        nframes = len(frame_ids)
        return nframes,True
    else:
        cframes = len(frame_ids)
        if nframes == cframes:
            return cframes,all_eq and True
        else:
            return nframes,False
    
def get_image_paths(ipath,nframes):

    # -- get total num frames --
    ipaths = list(glob.glob(str(ipath / "*")))
    tframes = len(ipaths)
    ref_id = tframes // 2
    if not(nframes > 0): nframes = tframes
    
    # -- create ref --
    fstart = ref_id - nframes//2
    frame_ids = np.arange(fstart,fstart+nframes)

    # -- load selected files --
    ipaths = []
    for t in range(nframes):
        fid = frame_ids[t]
        path = ipath / ("%05d.png" % fid)
        ipaths.append(path)
    
    return ipaths,frame_ids

def get_flow_paths(fdir,group,isize,ps,ipaths,frame_ids):
    
    # -- try reading flow --
    pstr = get_param_str(isize,ps)
    fpath = fdir/group/pstr
    nframes,K = len(ipaths),1
    ref_fid = "%02d" % (nframes//2)
    valid = check_valid_burst_nnf(group,0,nframes,fpath,K,ipaths,
                                  check_data=CHECK_NNF_DATA)

    # -- create nnfs if not existant --
    if not valid: 
        
        # -- read images --
        burst = read_burst(ipaths,isize)
        burst = rearrange(burst,'t c h w -> t h w c')

        # -- else create nnfs @ specified path --
        locs,vals = create_nnf_burst(burst,group,frame_ids,fpath,ps,K)

    # -- return list of valid nnfs --
    lpaths,vpaths = read_nnf_burst_paths(group,frame_ids,fpath,K)

    return lpaths,vpaths
