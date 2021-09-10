import os,cv2,glob,re,math
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tvF
from easydict import EasyDict as edict
import torch

VALIDATE_INDICES = dict()
VALIDATE_INDICES['2012'] = [0, 12, 15, 16, 17, 18, 24, 30, 38, 39, 42, 50, 54, 59, 60, 61, 77, 78, 81, 89, 97, 101, 107, 121, 124, 142, 145, 146, 152, 154, 155, 158, 159, 160, 164, 182, 183, 184, 190]
VALIDATE_INDICES['2015'] = [10, 11, 12, 25, 26, 30, 31, 40, 41, 42, 46, 52, 53, 72, 73, 74, 75, 76, 80, 81, 85, 86, 95, 96, 97, 98, 104, 116, 117, 120, 121, 126, 127, 153, 172, 175, 183, 184, 190, 199]

# ======== PLEASE MODIFY ========
kitti_root = r"/srv/disk3tb/home/gauenk/data/kitti/"
VERBOSE = True

def vprint(*args,**kwargs):
    if VERBOSE:
        print(*args,**kwargs)

def dir_to_burst_info(path_images):
    # -- get burst ids --
    burst_ids = []
    glob_path = str(Path(path_images) / "*")
    match_str = r"(?P<id>[0-9]{6})_(?P<t>[0-9]+)"
    for full_path in glob.glob(glob_path):
        stem = Path(full_path).stem
        match = re.match(match_str,stem).groupdict()
        group_id = match['id']
        # group_t = match['t']
        burst_ids.append(group_id)
    burst_ids = np.unique(burst_ids)

    # -- get burst for each frame --
    STANDARD_FRAMES = 21
    burst_info = {'ids':[],'nframes':[],'ref_t':[]}
    for burst_id in burst_ids:
        glob_path = str(Path(path_images) / Path("%s_*png" % burst_id))
        burst_info['ids'].append(burst_id)
        burst_info['nframes'].append(len(glob.glob(glob_path)))
        if burst_info['nframes'][-1] == STANDARD_FRAMES:
            burst_info['ref_t'].append(burst_info['nframes'][-1] // 2)
        else:
            nums = []
            for burst_t in glob.glob(glob_path):
                stem = Path(burst_t).stem
                match = re.match(match_str,stem).groupdict()
                group_t = int(match['t'])
                nums.append(group_t)
            nums = sorted(nums)
            T = len(nums)
            ref_t = nums[T//2]
            burst_info['ref_t'].append(ref_t)

    # -- to pandas --
    burst_info = pd.DataFrame(burst_info)
    burst_info = burst_info.set_index("ids")
    return burst_info


def read_dataset_sample(burst_id,ref_t,nframes,edition,
                        path = None, crop = None, resize = None,
                        nnf_K = 1, nnf_ps = 3):
    if path is None:
        path = kitti_path
    
    # -- setup paths --
    path_images = path[edition + 'image']
    path_flows = path[edition + 'flow_occ']
    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize)
    nnf_mid_path = Path("%s_%d" % (rs_nnf,nnf_ps))
    path_nnf = Path(path[edition + 'nnf']) / nnf_mid_path
    paths = edict({'images':path_images,'flows':path_flows,'nnf':path_nnf})

    # -- load image information --
    ref_path = os.path.join(path_images, '%s_%02d.png' % (burst_id, ref_t))
    ref_frame = tvF.to_tensor(Image.open(ref_path).convert("RGB"))
    frame_ids = np.arange(ref_t-nframes//2,ref_t+math.ceil(nframes/2))            

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
        inputs = [burst_id,ref_t,fid,paths,crop,resize,nnf_K]
        info = read_frame_info(*inputs)
        frame,nnf_loc,nnf_val,flow,occ = info

        # -- append to frame burst sample --
        burst.append(frame)
        nnf_locs.append(nnf_loc)
        nnf_vals.append(nnf_val)
        flows.append(flow)
        occs.append(occ)

    # -- concat results --
    results = {}
    results['burst'] = np.stack(burst)
    results['nnf_locs'] = np.stack(nnf_loc)
    results['nnf_vals'] = np.stack(nnf_vals)
    results['flows'] = np.stack(flows)
    results['occs'] = np.stack(occs)

    return results
    
        
def read_dataset_paths(path = None, editions = 'mixed', parts = 'mixed',
                 nframes = None, crop = None, resize = None,
                 nnf_K = 1, nnf_ps = 3, samples = None):

    if path is None:
        path = kitti_path

    dataset = dict()
    dataset['burst_id'] = []
    dataset['edition'] = []
    dataset['ref_t'] = []

    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize)
    nnf_mid_path = Path("%s_%d" % (rs_nnf,nnf_ps))
    editions = ('2012', '2015') if editions == 'mixed' else (editions, )
    for edition in editions:

        path_images = path[edition + 'image']
        path_flows = path[edition + 'flow_occ']
        path_nnf = Path(path[edition + 'nnf']) / nnf_mid_path
        paths = edict({'images':path_images,'flows':path_flows,'nnf':path_nnf})

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

            # -- extract reference frame from info --
            ref_t = int(burst_info.loc[burst_id]['ref_t'])

            # -- append dataset info --
            dataset['burst_id'].append(burst_id)
            dataset['edition'].append(edition)
            dataset['ref_t'].append(ref_t)
            dataset['nframes'].append(burst_nframes)

    return dataset

def read_frame_info(burst_id,ref_t,fid,paths,crop,resize,K):
    
    # -- load image --
    frame_path = os.path.join(paths.images, '%s_%s.png' % (burst_id, fid))
    img = cv2.cvtColor(cv2.imread(frame_path),cv2.COLOR_BGR2RGB)

    # -- read flow --
    flow_path = Path(os.path.join(paths.flows, '%s_10.png' % burst_id))
    flow_exists = flow_path.exists()
    if flow_exists:
        flow_occ = cv2.imread(str(flow_path), -1)
    else:
        flow_occ = np.nan * np.ones(img.shape)
        occ = np.nan * np.ones(img.shape)

    # -- read nnf --
    nnf_locs,nnf_vals = read_nnf(burst_id,ref_t,fid,paths.nnf,K)

    # -- crop --
    if crop is not None:
        img = img[-crop[0]:, :crop[1]]
        nnf_locs = nnf_locs[-crop[0]:, :crop[1]]
        nnf_vals = nnf_vals[-crop[0]:, :crop[1]]
        if flow_exists:
            flow_occ = flow_occ[-crop[0]:, :crop[1]]

    # -- reformat flows --
    if flow_exists:
        flow = np.flip(flow_occ[..., 1:3], axis=-1).astype(np.float32)
        flow = (flow - 32768.) / (64.)
        occ = flow_occ[..., 0:1].astype(np.uint8)

    # -- resize --
    if resize is not None:
        img = cv2.resize(img, resize)

        if flow_exists:
            tmp1 = (np.array(resize, dtype = np.float32) - 1.0)
            tmp2 = (np.array([flow.shape[d] for d in (1, 0)], dtype = np.float32) - 1.0)
            flow = cv2.resize(flow, resize) * (tmp1 / tmp2)[np.newaxis, np.newaxis, :]

            occ = cv2.resize(occ.astype(np.float32), resize)[..., np.newaxis]
            flow = flow / (occ + (occ == 0))
            occ = (occ * 255).astype(np.uint8)
    else:
        if flow_exists:
            occ = occ * 255

    return img,nnf_locs,nnf_vals,flow,occ

def read_nnf(burst_id,ref_t,fid,path_nnf,K):
    # -- load nnf --
    vals,locs = [],[]
    for k in range(K):
        loc_str = "loc_%s_%02d_%s_%02d.pt" % (burst_id,ref_t,fid,k)
        val_str = "val_%s_%02d_%s_%02d.pt" % (burst_id,ref_t,fid,k)
        path_nnf_loc = path_nnf / loc_str
        path_nnf_val = path_nnf / val_str
        vals_k = torch.load(path_nnf_val)
        locs_k = torch.load(path_nnf_loc)
        vals.append(vals_k)
        locs.append(locs_k)
    vals = np.stack(vals,axis=0)
    locs = np.stack(locs,axis=0)
    return vals,locs
    
def read_dataset_testing(path = None, editions = 'mixed', resize = None, samples = None):
    raise NotImplemented("Not implemented. Sorry!")

if __name__ == '__main__':
    dataset = read_dataset(resize = (1024, 436))
    # print(dataset['occ'][0].shape)
