"""

We read bursts of images from 
the long sequence of images
with length smaller than 
the original sequence


There not concept of 
optical flow in this dataset!

We only care about Nearest Neighbor Fields!

"""


# -- python imports --
import os,cv2,glob,re,math
import numpy as np
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict

# -- torch imports --
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
from datasets.kitti.nnf_io import read_nnf,write_nnf,check_nnf,get_nnf
from .utils import *    
        
def check_valid_burst_nnf(burst_id,fstart,nframes,path_nnf,nnf_K):
    ref_fid = '%02d' % int(fstart+nframes//2)
    frame_ids = np.arange(fstart,fstart+nframes)
    for t in range(nframes):
        fid = '%02d' % frame_ids[t]
        if check_nnf(burst_id,ref_fid,fid,path_nnf,nnf_K) is False:
            return False
    return True

def read_frame(paths,burst_id,fid):
    frame_path = Path(os.path.join(paths.images, '%s_%s.png' % (burst_id, fid)))
    if not frame_path.exists():
        raise IndexError(f"Frame {str(frame_path)} does not exist.")
    img = cv2.cvtColor(cv2.imread(str(frame_path)),cv2.COLOR_BGR2RGB)
    return img

def read_frame_info(burst_id,ref_fid,fid,paths,crop,resize,ref_frame,nnf_ps,nnf_K):
    
    # -- load image --
    img = read_frame(paths,burst_id,fid)

    # -- get that nnf --
    nnf_vals,nnf_locs = get_nnf(ref_frame,img,burst_id,ref_fid,
                                fid,paths.nnf,nnf_ps,nnf_K)

    # -- crop --
    if crop is not None:
        img = img[-crop[0]:, :crop[1]]
        nnf_vals = nnf_vals[-crop[0]:, :crop[1]]
        nnf_locs = nnf_locs[-crop[0]:, :crop[1]]

    # -- resize --
    if resize is not None:
        img = cv2.resize(img, resize)

    return img,nnf_vals,nnf_locs


def read_dataset_sample(burst_id,nframes,edition,fstart,istest,
                        path = None, crop = None, resize = None,
                        nnf_K = 1, nnf_ps = 3):
    if path is None:
        path = kitti_path
    
    # -- setup paths --
    if istest:
        path_images = path[edition + 'image' + 'test']
        path_nnf = Path(path[edition + 'nnf' + 'test'])
    else:
        path_images = path[edition + 'image']
        path_nnf = Path(path[edition + 'nnf'])

    # -- finish paths --
    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize)
    nnf_mid_path = Path("%s_%d" % (rs_nnf,nnf_ps))
    path_nnf = path_nnf / nnf_mid_path
    paths = edict({'images':path_images,'nnf':path_nnf})

    # -- init storage --
    burst = []
    nnf_vals = []
    nnf_locs = []

    # -- frame indexing --
    ref_t = fstart+nframes//2
    frame_ids = np.arange(fstart,fstart+nframes)

    # -- read reference for possible nnf compute --
    ref_fid = '%02d' % frame_ids[nframes//2]
    ref_frame = read_frame(paths,burst_id,ref_fid)

    # -- loop over frames --            
    for t in range(nframes):

        # -- load image and compute nnf --
        fid = '%02d' % frame_ids[t]
        
        # -- load frame burst info --
        inputs = [burst_id,ref_fid,fid,paths,crop,resize,ref_frame,nnf_ps,nnf_K]
        info = read_frame_info(*inputs)
        frame,nnf_val,nnf_loc = info

        # -- append to frame burst sample --
        burst.append(frame)
        nnf_vals.append(nnf_val)
        nnf_locs.append(nnf_loc)

    # -- concat results --
    results = {}
    results['burst'] = burst
    results['nnf_vals'] = np.stack(nnf_vals)
    results['nnf_locs'] = np.stack(nnf_locs)


    return results

def read_dataset_paths(path = None, editions = 'mixed', parts = 'mixed',
                 nframes = None, crop = None, resize = None,
                       nnf_K = 1, nnf_ps = 3, nnf_exists = True, samples = None):

    if path is None:
        path = kitti_path
    if nframes is None:
        raise ValueError("nframes must be a positive int.")

    dataset = dict()
    dataset['burst_id'] = []
    dataset['edition'] = []
    dataset['nframes'] = []
    dataset['fstart'] = []

    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize)
    nnf_mid_path = Path("%s_%d" % (rs_nnf,nnf_ps))
    editions = ('2012', '2015') if editions == 'mixed' else (editions, )
    for edition in editions:

        path_images = path[edition + 'image']
        path_nnf = Path(path[edition + 'nnf']) / nnf_mid_path
        paths = edict({'images':path_images,'nnf':path_nnf})

        num_files = len(os.listdir(path_images)) - 1
        ind_valids = VALIDATE_INDICES[edition]
        n_valids = len(ind_valids)
        valid_count = 0
        burst_info = dir_to_burst_info(path_images)
        
        # -- loop over burst images --
        burst_ids = burst_info.index.to_list()
        for burst_id in burst_ids:

            # -- split by dataset; skip if not correct selection --
            if valid_count < n_valids and ind_valids[valid_count] == burst_id:
                valid_count += 1
                if parts == 'train': continue
            elif parts == 'valid': continue

            # -- skip if sequence is too short --
            burst_nframes = int(burst_info.loc[burst_id]['nframes'])
            if burst_nframes < nframes:
                vprint(f"Skipping burst_id {burst_id} since not enough frames.")
                continue

            # -- append for each possible start position --
            for fstart in range(burst_nframes-nframes+1):

                # -- check if valid burst for nnf --
                if not check_valid_burst_nnf(burst_id,fstart,nframes,paths.nnf,nnf_K):
                    if nnf_exists: continue # require the nnf must exist

                # -- append --
                dataset['burst_id'].append(burst_id)
                dataset['edition'].append(edition)
                dataset['nframes'].append(burst_nframes)
                dataset['fstart'].append(fstart)

    return dataset


def read_dataset_testing(path = None, editions = 'mixed',
                         nframes = None, crop = None, resize = None,
                         nnf_K = 1, nnf_ps = 3, nnf_exists = True,
                         samples = None):

    if nframes is None:
        raise ValueError("nframes must be a positive int.")

    dataset = dict()
    dataset['burst_id'] = []
    dataset['edition'] = []
    dataset['nframes'] = []
    dataset['fstart'] = []

    editions = ('2012', '2015') if editions == 'mixed' else (editions, )
    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize)
    nnf_mid_path = Path("%s_%d" % (rs_nnf,nnf_ps))
    for edition in editions:
        path_images = path[edition + 'image' + 'test']
        path_nnf = Path(path[edition + 'nnf' + 'test']) / nnf_mid_path
        num_files = (len(os.listdir(path_images)) - 10) // 21
        burst_info = dir_to_burst_info(path_images)
        
        # -- loop over burst images --
        burst_ids = burst_info.index.to_list()
        for burst_id in burst_ids:

            # -- skip if sequence is too short --
            burst_nframes = int(burst_info.loc[burst_id]['nframes'])
            if burst_nframes < nframes:
                vprint(f"Skipping burst_id {burst_id} since not enough frames.")
                continue

            # -- append for each possible start position --
            for fstart in range(burst_nframes-nframes+1):

                # -- check if valid burst for nnf --
                if not check_valid_burst_nnf(burst_id,fstart,nframes,path_nnf,nnf_K):
                    if nnf_exists: continue # require the nnf must exist

                dataset['burst_id'].append(burst_id)
                dataset['edition'].append(edition)
                dataset['nframes'].append(burst_nframes)
                dataset['fstart'].append(fstart)

    return dataset


if __name__ == '__main__':
    dataset = read_dataset(resize = (1024, 436))
    # print(dataset['occ'][0].shape)
