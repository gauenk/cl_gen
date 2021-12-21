
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
from datasets.nnf_io import create_nnf_burst,check_valid_burst_nnf,read_nnf_burst_paths
from datasets.nnf_io import read_nnf_burst,read_nnf

# -- cache info --
import cache_io


CHECK_NNF_DATA = False # make true for nnf checking; false for faster checking
CHECK_VALID = True

def read_image_size(ipaths):
    height,width = 10000,100000
    for ipath in ipaths:
        _width,_height = Image.open(ipath).size
        if _height < height:
            height = _height
        if _width < width:
            width = _width
    return height,width

def read_icrops(ipaths,isize):
    height,width = read_image_size(ipaths)
    i_height,i_width = isize
    nheight = height // i_height
    nwidth = width // i_width
    assert (nheight>0) and (nwidth>0)

    heights = [i*i_height for i in range(nheight)]
    widths = [i*i_width for i in range(nwidth)]
    crops = np.meshgrid(heights,widths)
    crops = np.stack(crops).reshape(2,-1).T
    return crops

def read_split_ids(sdir,split):

    # -- get filename --
    if split == "train":
        fn = "train.txt"
    elif split == "val":
        fn = "val.txt"
    elif split == "test" or split == "test-dev":
        fn = "test-dev.txt"

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

def read_burst(ipaths,isize,icrop):
    dyn_clean = []
    for ipath in ipaths:
        img = Image.open(ipath).convert("RGB")
        dyn_clean.append(tvF.to_tensor(img))
    dyn_clean = torch.stack(dyn_clean)
    if not(isize is None):
        if icrop[0] == -1 and icrop[1] == -1:
            dyn_clean = tvF.center_crop(dyn_clean,isize)
        else:
            hslice = slice(icrop[0],icrop[0]+isize[0])
            wslice = slice(icrop[1],icrop[1]+isize[1])
            dyn_clean = dyn_clean[...,hslice,wslice]
    return dyn_clean

def read_pix(fpaths,vpaths=None):

    # -- read all locs --
    _,locs = read_nnf_burst(vpaths,fpaths)
    locs = rearrange(locs,'t k h w two -> two t k h w')
    # locs = np.flip(locs,axis=(0,))
    # ref_flows = locs2flow(locs)
    ref_flows = locs
    return ref_flows

def pix2flow(pix):
    two,t,k,h,w = pix.shape
    ref_frame = t//2
    coords = repeat(pix[:,ref_frame],'two k h w -> two t k h w',t=t)
    flow = coords - pix
    flow[0] = -flow[0]
    return flow

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
    for group in split_groups:
        # if not(group in split_groups): continue
        ipaths,frame_ids = get_image_paths(idir/group,nframes)
        paths['images'][group] = ipaths
        paths['frame_ids'][group] = frame_ids
        out_nframes,all_eq = update_nframes(out_nframes,frame_ids,all_eq)

    # -- read flow paths --
    for group in split_groups:
        # if not(group in split_groups): continue
        ipaths = paths['images'][group]
        frame_ids = paths['frame_ids'][group]
        vpaths,fpaths = get_flow_paths(fdir,group,isize,ps,ipaths,frame_ids,['c','c'])
        paths['vals'][group] = vpaths
        paths['flows'][group] = fpaths

    return paths,out_nframes,all_eq

def create_cache_config(fdir,sdir,split,isize,ps,nframes):
    config = {}
    config['fdir'] = str(fdir)
    config['sdir'] = str(sdir)
    config['split'] = split
    if not(isize is None):
        config['isize_0'] = isize[0]
        config['isize_1'] = isize[1]
    else:
        config['isize_0'] = None
        config['isize_1'] = None
    config['ps'] = ps
    config['nframes'] = nframes
    return config

def read_subburst_files(idir,fdir,sdir,split,isize,ps,nframes):
    """
    We take all the standard files and then

    """

    # -- try from cache --
    cache_root,cache_name = idir/".cache/","v1"
    cache = cache_io.ExpCache(cache_root,cache_name)
    config = create_cache_config(fdir,sdir,split,isize,ps,nframes)
    # cache.clear()
    results = cache.load_exp(config)
    if not(results is None):
        spaths = results['spaths']
        out_nframes = results['out_nframes']
        all_eq = results['all_eq']
        return spaths,out_nframes,all_eq
    uuid = cache.get_uuid(config)

    # -- we read all frames --
    global CHECK_VALID
    CHECK_VALID = False
    paths,out_nframes,all_eq = read_files(idir,fdir,sdir,split,isize,ps,-1)
    CHECK_VALID = True

    # -- init output var --
    spaths = edict({'images':{},'flows':{},'crops':{},
                   'vals':{},'frame_ids':{}})

    # -- we take each subset of "nframes" from each burst
    groups = list(paths['images'].keys())
    for group in groups:
        print(group)

        # -- take frame range for each step --
        tframes = len(paths['images'][group])
        for fstart in range(tframes-nframes):
            print(fstart)

            # -- primary logic does not depend on image patches  --
            ref_frame = fstart + nframes//2
            index = slice(fstart,fstart+nframes)
            ipaths = paths['images'][group][index]
            fpaths = paths['flows'][group][index]
            vpaths = paths['vals'][group][index]
            frame_ids = [fstart + i for i in range(nframes)]


            # -- take frame for each patch --
            icrops = read_icrops(ipaths,isize)
            ncrops = len(icrops)
            for crop_idx in range(ncrops):

                # -- modify the fpaths and vpaths --
                icrop = icrops[crop_idx]
                vpaths,fpaths = get_flow_paths(fdir,group,isize,ps,
                                               ipaths,frame_ids,icrop)

                # -- "sgroup" defines how many samples our dataset contains --
                start_frame,start_height,start_width = fstart,icrop[0],icrop[1]
                start_str = "-%d-%d-%d" % (start_frame,start_height,start_width)
                sgroup = group + start_str

                # -- fill in results --
                spaths['images'][sgroup] = ipaths
                spaths['flows'][sgroup] = fpaths
                spaths['vals'][sgroup] = vpaths
                spaths['frame_ids'][sgroup] = frame_ids
                spaths['crops'][sgroup] = icrop

    # -- save results to cache output --
    results = {'spaths':spaths,'out_nframes':out_nframes,'all_eq':all_eq}
    cache.save_exp(uuid,config,results)

    return spaths,out_nframes,all_eq

def replace_paths(paths,fidx,fstart,nframes):
    full_path = paths[fidx][0]
    path_stem = full_path.stem
    full_path = str(full_path)
    split_path_stem = path_stem.split("_")
    split_path_stem[-3] = "%02d" % (fstart + nframes//2)
    path_stem_f = "_".join(split_path_stem)
    paths[fidx][0] = Path(full_path.replace(path_stem,path_stem_f))
    exit()

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
        path = ipath / ("%05d.jpg" % fid)
        ipaths.append(path)

    return ipaths,frame_ids

def get_flow_paths(fdir,group,isize,ps,ipaths,frame_ids,icrop):

    # -- try reading flow --
    pstr = get_param_str(isize,ps)
    fpath = fdir/group/pstr
    nframes,K = len(ipaths),1
    ref_fid = "%02d" % (frame_ids[nframes//2])

    # -- decide if we check valid --
    if CHECK_VALID:
        valid = check_valid_burst_nnf(group,frame_ids,fpath,K,isize,icrop,
                                      ipaths,check_data=CHECK_NNF_DATA)
    else:
        valid = True

    # -- create nnfs if not existant --
    if not valid:

        # -- read images --
        burst = read_burst(ipaths,isize,icrop)
        burst = rearrange(burst,'t c h w -> t h w c')

        # -- else create nnfs @ specified path --
        vals,locs = create_nnf_burst(burst,group,frame_ids,fpath,ps,K,icrop,isize)

    # -- return list of valid nnfs --
    vpaths,lpaths = read_nnf_burst_paths(group,frame_ids,fpath,K,icrop)

    return vpaths,lpaths
