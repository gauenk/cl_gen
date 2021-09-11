import os,cv2,glob,re,math
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tvF
from easydict import EasyDict as edict
import torch
from datasets.kitti.nnf_io import get_nnf,check_valid_burst_nnf
from .utils import *

CHECK_NNF_DATA=False

def read_frame_info(burst_id,ref_fid,fid,paths,crop,resize,ref_frame,nnf_ps,nnf_K):
    
    # -- load image --
    img = read_frame(paths.images,burst_id,fid)
    assert img.shape[-1] == 3,"Color is on third dim and its 3."
    
    # -- read flow --
    flow_path = Path(os.path.join(paths.flows, '%s_10.png' % burst_id))
    flow_exists = flow_path.exists()
    if flow_exists:
        flow_occ = cv2.imread(str(flow_path), -1)
    else:
        flow = np.nan * np.ones(img.shape)
        occ = np.nan * np.ones(img.shape)

    # -- read nnf --
    nnf_vals,nnf_locs = get_nnf(ref_frame,img,burst_id,ref_fid,fid,paths.nnf,nnf_ps,nnf_K)

    # -- crop --
    if crop is not None:
        img = img[-crop[0]:, :crop[1]]
        nnf_vals = nnf_vals[-crop[0]:, :crop[1]]
        nnf_locs = nnf_locs[-crop[0]:, :crop[1]]
        if flow_exists:
            flow_occ = flow_occ[-crop[0]:, :crop[1]]

    # -- reformat flows --
    if flow_exists:
        flow = np.flip(flow_occ[..., 1:3], axis=-1).astype(np.float32)
        flow = (flow - 32768.) / (64.)
        occ = flow_occ[..., 0:1].astype(np.uint8)

    # -- resize --
    if resize is not None:

        # -- resize image --
        img = cv2.resize(img, resize)

        # -- resize helpers --
        tmp1 = (np.array(resize, dtype = np.float32) - 1.0)
        tmp2 = (np.array([flow.shape[d] for d in (1, 0)], dtype = np.float32) - 1.0)

        # -- resize nnf --
        nnf_locs = nnf_locs.astype(np.float64)
        nnf_locs = list(nnf_locs)
        for t in range(len(nnf_locs)):
            nnf_locs[t] = cv2.resize(nnf_locs[t], resize) 
            nnf_locs[t] = nnf_locs[t]* (tmp1 / tmp2)[np.newaxis, np.newaxis, :]
            nnf_locs[t] = np.round(nnf_locs[t]).astype(np.int64)
        nnf_locs = np.stack(nnf_locs)

        # -- resize flow --
        if flow_exists:
            flow = cv2.resize(flow, resize) * (tmp1 / tmp2)[np.newaxis, np.newaxis, :]
            occ = cv2.resize(occ.astype(np.float32), resize)[..., np.newaxis]
            flow = flow / (occ + (occ == 0))
            occ = (occ * 255).astype(np.uint8)
    else:
        if flow_exists:
            occ = occ * 255

    return img,nnf_vals,nnf_locs,flow,occ

def read_dataset_sample(burst_id,ref_t,nframes,edition,istest,
                        path = None, crop = None, resizes = None,
                        nnf_K = 1, nnf_ps = 3):
    if path is None:
        path = kitti_path
    
    # -- setup paths --
    if istest:
        path_images = path[edition + 'image' + 'test']
        path_nnf = Path(path[edition + 'nnf' + 'test'])
        path_flows = Path("kitti_test_flow_dne")
    else:
        path_images = path[edition + 'image']
        path_nnf = Path(path[edition + 'nnf'])
        path_flows = path[edition + 'flow_occ']

    # -- finish paths --
    resize = resizes.path
    resize_hw = (resize[1],resize[0])
    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize_hw)
    nnf_mid_path = Path("%s_%d" % (rs_nnf,nnf_ps))
    path_nnf = path_nnf / nnf_mid_path
    paths = edict({'images':path_images,'flows':path_flows,'nnf':path_nnf})
    resize = resizes.load

    # -- load image information --
    ref_fid = '%02d' % ref_t
    ref_frame = read_frame(paths.images,burst_id,ref_fid)
    fstart = ref_t - nframes//2
    frame_ids = np.arange(fstart,fstart+nframes)

    # -- init storage --
    burst = []
    nnf_locs = []
    nnf_vals = []
    flows = []
    occs = []

    # -- loop over frames --            
    for t in range(nframes):

        # -- load image and compute nnf --
        fid = '%02d' % frame_ids[t]
        
        # -- load frame burst info --
        inputs = [burst_id,ref_fid,fid,paths,crop,resize,ref_frame,nnf_ps,nnf_K]
        info = read_frame_info(*inputs)
        frame,nnf_val,nnf_loc,flow,occ = info

        # -- append to frame burst sample --
        burst.append(frame)
        nnf_vals.append(nnf_val)
        nnf_locs.append(nnf_loc)
        flows.append(flow)
        occs.append(occ)

    # -- concat results --
    results = {}
    results['burst'] = burst
    results['ref_frame'] = ref_frame
    results['nnf_vals'] = np.stack(nnf_vals)
    results['nnf_locs'] = np.stack(nnf_locs)
    results['flows'] = np.stack(flows)
    results['occs'] = np.stack(occs)

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
    dataset['ref_t'] = []
    dataset['nframes'] = []

    resize_hw = (resize[1],resize[0])
    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize_hw)
    nnf_mid_path = Path("%s_%d" % (rs_nnf,nnf_ps))
    editions = ('2012', '2015') if editions == 'mixed' else (editions, )
    for edition in editions:

        path_images = path[edition + 'image']
        path_flows = path[edition + 'flow_occ']
        path_nnf = Path(path[edition + 'nnf']) / nnf_mid_path
        paths = edict({'images':path_images,'flows':path_flows,'nnf':path_nnf})

        num_files = len(os.listdir(paths.images)) - 1
        ind_valids = VALIDATE_INDICES[edition]
        n_valids = len(ind_valids)
        valid_count = 0
        burst_info = dir_to_burst_info(paths.images)
        
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

            # -- extract reference frame from info --
            ref_t = int(burst_info.loc[burst_id]['ref_t'])
            
            # -- check if valid burst for nnf --
            fstart = ref_t - nframes//2
            if not check_valid_burst_nnf(burst_id,fstart,nframes,paths,
                                         nnf_K,CHECK_NNF_DATA):
                if nnf_exists: continue # require the nnf must exist

            # -- append dataset info --
            dataset['burst_id'].append(burst_id)
            dataset['edition'].append(edition)
            dataset['ref_t'].append(ref_t)
            dataset['nframes'].append(burst_nframes)

    return dataset


def read_dataset_testing(path = None, editions = 'mixed',
                         nframes = None, crop = None, resize = None,
                         nnf_K = 1, nnf_ps = 3, nnf_exists = True, samples = None):

    if nframes is None:
        raise ValueError("nframes must be a positive int.")

    dataset = dict()
    dataset['burst_id'] = []
    dataset['edition'] = []
    dataset['ref_t'] = []
    dataset['nframes'] = []

    editions = ('2012', '2015') if editions == 'mixed' else (editions, )
    resize_hw = (resize[1],resize[0])
    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize_hw)
    nnf_mid_path = Path("%s_%d" % (rs_nnf,nnf_ps))
    for edition in editions:
        path_images = path[edition + 'image' + 'test']
        path_nnf = Path(path[edition + 'nnf' + 'test']) / nnf_mid_path
        num_files = (len(os.listdir(path_images)) - 10) // 21
        burst_info = dir_to_burst_info(path_images)
        paths = edict({'images':path_images,'nnf':path_nnf})
        
        # -- loop over burst images --
        burst_ids = burst_info.index.to_list()
        for burst_id in burst_ids:

            # -- skip if sequence is too short --
            burst_nframes = int(burst_info.loc[burst_id]['nframes'])
            if burst_nframes < nframes:
                vprint(f"Skipping burst_id {burst_id} since not enough frames.")
                continue

            # -- extract reference frame from info --
            ref_t = int(burst_info.loc[burst_id]['ref_t'])

            # -- check if valid burst for nnf --
            fstart = ref_t - nframes//2
            if not check_valid_burst_nnf(burst_id,fstart,nframes,paths,
                                         nnf_K,CHECK_NNF_DATA):
                if nnf_exists: continue # require the nnf must exist

            # -- append dataset info --
            dataset['burst_id'].append(burst_id)
            dataset['edition'].append(edition)
            dataset['ref_t'].append(ref_t)
            dataset['nframes'].append(burst_nframes)

    return dataset


if __name__ == '__main__':
    dataset = read_dataset(resize = (1024, 436))
    # print(dataset['occ'][0].shape)
