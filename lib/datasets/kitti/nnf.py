"""
Create Nearest Neighbor Fields (NNF) annotations.

"""

# -- python --
import glob,re,os,cv2,math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy.random as npr
from einops import rearrange,repeat
from PIL import Image
from easydict import EasyDict as edict

# -- pytorch --
import torch
import torchvision.transforms.functional as tvF

# -- faiss imports --
# import faiss

# -- project --
from pyutils import tile_patches,flow_to_color,save_image
from align.nnf import compute_nnf
from datasets.kitti.paths import get_kitti_path

def check_nnf_dataset_exists(path_nnf):
    if not path_nnf.exists(): path_nnf.mkdir(parents=True)
    return False

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

# def compute_nnf(ref_img,prop_img,patchsize,K=10,gpuid=0):
#     """
#     Compute the Nearest Neighbor Field for Optical Flow
#     """
#     C,H,W = ref_img.shape

#     # -- tile patches --
#     query = repeat(prop_img,'c h w -> 1 1 c h w')
#     q_patches = tile_patches(query,patchsize).pix
#     B,N,R,ND = q_patches.shape
#     query = rearrange(q_patches,'b n r nd -> (b n r) nd').numpy()

#     db = repeat(ref_img,'c h w -> 1 1 c h w')
#     db_patches = tile_patches(db,patchsize).pix
#     Bd,Nd,Rd,NDd = db_patches.shape
#     database = rearrange(db_patches,'b n r nd -> (b n r) nd').numpy()

#     # -- faiss setup --
#     res = faiss.StandardGpuResources()
#     faiss_cfg = faiss.GpuIndexFlatConfig()
#     faiss_cfg.useFloat16 = False
#     faiss_cfg.device = gpuid

#     # -- create database --
#     gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
#     gpu_index.add(database)
    
#     # -- execute search --
#     D,I = gpu_index.search(query,K)

#     # -- get nnf (x,y) from I --
#     S,Kp = I.shape
#     locs = np.unravel_index(I,(H,W))
#     locs = np.stack(locs,axis=-1)
#     locs = rearrange(locs,'(h w) k two -> h w k two',h=H)
#     vals = rearrange(D,'(h w) k -> h w k',h=H)
#     return vals,locs

def visualize_nnf(fn,vals,locs):

    # -- compute index grids --
    H,W = locs.shape[:2]
    hgrid = repeat(np.arange(H),'h -> h w',w=W)
    wgrid = repeat(np.arange(W),'w -> h w',h=H)
    index = np.stack([hgrid,wgrid],axis=-1)
    
    # -- compute delta --
    k,epsilon = 0,1e-8
    delta = locs[:,:,k] - index

    # -- create flow image --
    vis = flow_to_color(delta)
    vis = rearrange(torch.FloatTensor(vis)/255.,'h w c -> c h w')
    save_image(vis,fn)

def check_frame_nnf_exists(path_nnf,burst_id,fid,ref_t,K):
    for k in range(K):
        loc_str = "loc_%s_%02d_%s_%02d.pt" % (burst_id,ref_t,fid,k)
        val_str = "val_%s_%02d_%s_%02d.pt" % (burst_id,ref_t,fid,k)
        path_nnf_loc = path_nnf / loc_str
        path_nnf_val = path_nnf / val_str
        if not(path_nnf_loc.exists() and path_nnf_val.exists()):
            return False
    return True

def create_nnf_dataset(path, patchsize, resize, K=10,
                       editions = 'mixed', parts = 'mixed'):
    ps = patchsize
    rs_nnf = 'default' if resize is None else '{:d}_{:d}'.format(*resize)
    nnf_mid_path = Path("%s_%d" % (rs_nnf,ps))
    editions = ('2012', '2015') if editions == 'mixed' else (editions, )
    for edition in tqdm(editions):
        path_images = path[edition + 'image']
        burst_info = dir_to_burst_info(path_images)
        path_nnf = Path(path[edition + 'nnf']) / nnf_mid_path
        if check_nnf_dataset_exists(path_nnf): continue
        num_files = len(os.listdir(path_images)) - 1
        
        # -- loop over burst images --
        burst_ids = burst_info.index.to_list()
        for burst_id in tqdm(burst_ids,leave=False):

            # -- load burst reference --
            nframes = int(burst_info.loc[burst_id]['nframes'])
            ref_t = int(burst_info.loc[burst_id]['ref_t'])
            frame_ids = np.arange(ref_t-nframes//2,ref_t+math.ceil(nframes/2))
            ref_path = os.path.join(path_images, '%s_%02d.png' % (burst_id, ref_t))
            ref_img  = tvF.to_tensor(Image.open(ref_path).convert("RGB"))

            # -- loop over frames --            
            for t in tqdm(range(nframes),leave=False):

                # -- get frame id --
                fid = '%02d' % frame_ids[t]

                # -- check if files exist --
                if check_frame_nnf_exists(path_nnf,burst_id,fid,ref_t,K): continue

                # -- load image and compute nnf --
                frame_path = os.path.join(path_images, '%s_%s.png' % (burst_id, fid))
                img = tvF.to_tensor(Image.open(frame_path).convert("RGB"))
                vals,locs = compute_nnf(ref_img,img,patchsize)
                # fn = "./test_nnf_vis.png"
                # visualize_nnf(fn,vals,locs)

                # -- path for nnf --
                for k in range(K):
                    loc_str = "loc_%s_%02d_%s_%02d.pt" % (burst_id,ref_t,fid,k)
                    val_str = "val_%s_%02d_%s_%02d.pt" % (burst_id,ref_t,fid,k)
                    path_nnf_loc = path_nnf / loc_str
                    path_nnf_val = path_nnf / val_str
                    vals_k,locs_k = vals[:,:,k],locs[:,:,k]
                    torch.save(vals_k,path_nnf_val)
                    torch.save(locs_k,path_nnf_loc)

def create_kitti_nnf(root,patchsize,resize,K):
    path = get_kitti_path(root)
    create_nnf_dataset(path,patchsize,resize,K,'mixed','mixed')
