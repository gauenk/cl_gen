"""
Code to search for the aligned patches from bursts

"""

# -- python imports --
import numpy as np
from einops import rearrange,repeat
from itertools import chain, combinations
from pathlib import Path
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils import images_to_psnrs
from .tile_utils import *
from .abps_utils import *

def abi_global_search(cfg,burst,clean_imgs=None):

    # -- create vars for search --
    N,B,C,H,W = burst.shape
    PS = cfg.patchsize
    NH = cfg.nh_size

    # -- setup for best_indices --
    tiled = tile_bursts_global_motion(burst,PS,NH)
    if not (clean_imgs is None): clean_tiled = tile_bursts_global_motion(clean_imgs,PS,NH)

    # -- run search --
    scores,indices = run_abi_image_search_global_dynamics(tiled,PS,NH,clean_tiled)
    indices.cuda(non_blocking=True)

    # -- recover aligned burst images --
    aligned = aligned_images_from_global_dynamic_indices(tiled,indices)
    if not (clean_imgs is None):
        aligned = aligned_images_from_global_dynamic_indices(clean_tiled,indices)

    return aligned

def aligned_images_from_global_dynamic_indices(tiled,indices):
    B,N = indices.shape[:2]
    aligned = []
    for b in range(B):
        aligned_n = []
        for n in range(N):
            image = tiled[b,n,indices[b,n]]
            aligned_n.append(image)
        aligned_n = torch.stack(aligned_n,dim=0)
        aligned.append(aligned_n)
    aligned = torch.stack(aligned,dim=1)
    return aligned
    
def run_abi_image_search_global_dynamics(tiled,PS,NH,clean_tiled=None):

    # -- shapes --
    B,N,NH2 = tiled.shape[:3]
    REF_NH = get_ref_nh(NH)

    # -- init -- 
    best_scores = torch.zeros(B,N)
    best_indices = torch.zeros(B,N).type(torch.long)
    burst_indices = np.r_[np.r_[:N//2],np.r_[N//2+1:N]]
    best_indices[:,N//2] = REF_NH # default for middle frame

    # -- nh grids --
    nh_grids = create_powerset_pair_grids(NH2)
    print(f"[NH Grid]: {len(nh_grids)}")
    np.random.shuffle(nh_grids)
    nh_grids = nh_grids#[:1000]

    # -- [ testing only ] --
    best_psnrs = torch.zeros(B,N)
    best_psnr_indices = torch.zeros(B,N).type(torch.long)

    # -- run comparisons --
    for b in range(B):
        ref = repeat(tiled[b,N//2,REF_NH],'c h w -> nh2 c h w',nh2=NH2)
        if not (clean_tiled is None):
            clean_ref = repeat(clean_tiled[b,N//2,REF_NH],'c h w -> nh2 c h w',nh2=NH2)
        for n in range(N):
            print(f"n = {n}")
            if n == N//2: continue
            frames = tiled[b,n,:]
            best_score,best_index = best_global_image_arrangement_score(ref,frames,nh_grids)
            best_scores[b,n] = best_score
            best_indices[b,n] = best_index

            # -- [testing only] --
            if not (clean_tiled is None):
                clean_frames = clean_tiled[b,n,:]
                best_psnr,best_psnr_index = best_global_image_arrangement_psnr(clean_ref,clean_frames)
                best_psnrs[b,n] = best_psnr
                best_psnr_indices[b,n] = best_psnr_index
            if b == 0 and n == 0:
                tv_utils.save_image(ref,"ref_0.png",normalize=True)
                tv_utils.save_image(frames,"frames_00.png",normalize=True)

            print(f"Index_s v.s. Index_p: {best_indices[b,n]} v.s. {best_psnr_indices[b,n]}")

    # -- [ testing only ] --
    if not (clean_tiled is None):
        plot_score_and_psnr_info(best_scores,best_indices,best_psnrs,best_psnr_indices)

    return best_scores,best_indices

def plot_score_and_psnr_info(best_scores,best_indices,best_psnrs,best_psnr_indices):
    print("best_scores",best_scores)
    print("best_psnrs",best_psnrs)
    
def compute_score(frames,grid0,grid1,NH2):
    ave0 = torch.mean(frames[grid0],dim=0)
    ave1 = torch.mean(frames[grid1],dim=0)
    mse = F.mse_loss(ave0,ave1)
    score = torch.zeros(NH2,device=frames.device)
    count = torch.zeros(NH2,device=frames.device)
    score[grid0] += mse
    score[grid1] += mse
    count[grid0] += 1
    count[grid1] += 1
    return score,count

def best_global_image_arrangement_score(ref,frames,nh_grids):

    # -- crop images --
    NH2,C,H,W = frames.shape
    NH = int(np.sqrt(NH2))
    top,left = NH//2,NH//2
    crop_ref = tvF.crop(ref[[0]],top,left,H-NH//2,W-NH//2)
    crop_frames = tvF.crop(frames,top,left,H-NH//2,W-NH//2)
    aggregate = torch.cat([crop_ref,crop_frames],dim=0)

    # -- compute score --
    scores,counts = torch.zeros(NH2,device=ref.device),torch.zeros(NH2,device=ref.device)
    for nh_grid0,nh_grid1 in nh_grids:
        score,count = compute_score(aggregate,nh_grid0,nh_grid1,NH2)
        scores += score
        counts += count
    scores /= counts

    # -- find best --
    best_index = torch.argmin(scores)
    best_score = scores[best_index]

    return best_score,best_index


def best_global_image_arrangement_psnr(ref,frames):

    # -- crop images --
    NH2,C,H,W = frames.shape
    NH = int(np.sqrt(NH2))
    top,left = 0,0#NH//2,NH//2
    crop_ref = tvF.crop(ref,top,left,H,W)
    crop_frames = tvF.crop(frames,top,left,H,W)

    # -- compute score --
    scores = torch.FloatTensor(images_to_psnrs(crop_ref, crop_frames))
    # scores = F.mse_loss( crop_ref, crop_frames, reduction='none').reshape(NH2,-1)
    # scores = torch.mean(scores,dim=1)

    # -- find best --
    best_index = torch.argmax(scores)
    best_score = scores[best_index]

    return best_score,best_index
