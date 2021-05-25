"""
Code to search for the aligned patches from bursts

"""

# -- python imports --
import numpy as np
from einops import rearrange,repeat
from itertools import chain, combinations
from pathlib import Path
from easydict import EasyDict as edict

# from torch.multiprocessing import Pool, Process, set_start_method, Manager
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

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
from .abi_global_search import abi_global_search

def abp_search(cfg,burst):
    return abp_global_search(cfg,burst)

def abp_global_search(cfg,burst):

    # -- create vars for search --
    N,B,C,H,W = burst.shape
    PS = cfg.patchsize
    NH = cfg.nh_size

    # -- setup for best_indices --
    patches = tile_burst_patches(burst,PS,NH)

    # -- run search --
    # indices = run_abp_patch_search_pairs_global_dynamics(patches,PS,NH)
    scores,indices = run_abp_patch_search_global_dynamics(patches,PS,NH)
    indices.cuda(non_blocking=True)

    # -- recover aligned burst images --
    aligned = aligned_burst_image_from_indices_global_dynamics(patches,np.arange(N),indices)
    
    return scores,aligned
    

def run_abp_patch_search_global_dynamics(patches,PS,NH):

    # -- shapes --
    R,B,N = patches.shape[:3]
    REF_NH = get_ref_nh(NH)

    # -- init -- 
    best_scores = torch.zeros(B)
    best_indices = torch.zeros(B,N).type(torch.long)

    # -- run comparisons --
    for b in range(B):
        burst_indices = np.r_[np.r_[:N//2],np.r_[N//2+1:N]]
        args = [patches[:,[b]],burst_indices,PS,NH]
        scores_b,indices_b = abp_search_global_dynamics(*args)
        best_score_b,best_indices_b = scores_b[0],indices_b[0]
        best_indices_b = insert_nh_middle(best_indices_b,NH,burst_indices.shape[0])
        best_scores[b] = best_score_b
        best_indices[b] = best_indices_b
    return best_scores,best_indices

#
# Primary Coordinate Desc. Search Function
# 

def mp_abp_search_global_dynamics(procnum,returns,patches,burst_indices,PS,NH,K=2,verbose=False):
    scores,patches = abp_search_global_dynamics(patches,burst_indices,PS,NH,K=K,verbose=verbose)
    returns[procnum] = [scores,patches]

def abp_search_global_dynamics(patches,burst_indices,PS,NH,K=2,verbose=False):
    
    if burst_indices.shape[0] <= 2:
        return abp_search_exhaustive_global_dynamics(patches,burst_indices,PS,NH,K=K)


    # -- init shapes --
    R,B,N = patches.shape[:3]
    FMAX = np.finfo(np.float).max
    REF_NH = get_ref_nh(NH)
    BI = burst_indices.shape[0]
    H = int(np.sqrt(patches.shape[0]))
    
    # -- split the burst grid up --
    split_grids = create_split_burst_grid(N)
    S = split_grids.shape[0]

    #
    # -- recurse --
    #

    agg = edict()
    agg.scores = edict({str(n):[] for n in range(N)})
    agg.indices = edict({str(n):[] for n in range(N)})

    for s,burst_grid in enumerate(split_grids):
        scores,indices = abp_search_global_dynamics(patches,burst_grid,PS,NH,K=K)

        # -- append to aggregate --
        for i,n_int in enumerate(burst_grid):
            n = str(n_int)
            agg.scores[n].extend(scores)
            agg.indices[n].extend(indices[:,i])

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    # -- multiprocessing code --
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # manager = Manager()
    # returns = manager.dict()
    # tasks = []
    # results = []
    # for s,burst_grid in enumerate(split_grids):
    #     task = Process(target=mp_abp_search_global_dynamics,args=(s,returns,patches,burst_grid,PS,NH,),kwargs=dict(K=K))
    #     tasks.append(task)

    # -- execute tasks --
    # for t,task in enumerate(tasks):
    #     task.start()
    #     if t > 0 and t % 3 == 0: task.join()

    # -- convert to numpy --
    for n_int in range(N):
        n = str(n_int)
        agg.scores[n] = np.array(agg.scores[n])
        agg.indices[n] = np.array(agg.indices[n])

    # -- pick top K --
    top_nh_indices = []
    for n_int in range(N):
        if n_int == N//2: continue
        n = str(n_int)
        search_topK = np.argsort(agg.scores[n])
        nh_indices_topK = agg.indices[n][search_topK]
        top_nh_indices.append(np.array(nh_indices_topK))

    # -- create global search grid --
    top_nh_grid = create_grid_from_ndarrays(top_nh_indices)
    if verbose: print(f"Top NH Grid {len(top_nh_grid)}")

    # -- final global search --
    scores,indices = abp_search_exhaustive_global_dynamics(patches,burst_indices,PS,NH,
                                                                K=K,nh_grids=top_nh_grid)

    return scores,indices


def abp_search_exhaustive_global_dynamics(patches,burst_indices,PS,NH,
                                          K=-1,nh_grids=None,verbose=True):

    # -- init vars --
    R,B,N = patches.shape[:3]
    FMAX = np.finfo(np.float).max
    REF_NH = get_ref_nh(NH)
    if verbose: print(f"REF_NH: {REF_NH}")
    BI = burst_indices.shape[0]
    H = int(np.sqrt(patches.shape[0]))
    ref_patch = patches[:,:,[N//2],[REF_NH],:,:,:]

    # -- create search grids --
    if nh_grids is None: nh_grids = create_nh_grids(BI,NH)
    n_grids = create_n_grids(BI)
    if verbose: print(f"NH_GRIDS {len(nh_grids)} | N_GRIDS {len(n_grids)}")
    
    # -- randomly initialize grids --
    # np.random.shuffle(nh_grids)
    np.random.shuffle(n_grids)

    # -- init loop vars --
    scores = np.zeros(len(nh_grids))
    best_score,best_select = FMAX,None

    # -- remove boundary --
    subR = torch.arange(H*H-2*NH*H)+NH*H
    search = patches[subR]
    ref_patch = ref_patch[subR]

    # -- coordinate descent --
    for nh_index,nh_grid in enumerate(nh_grids):
        # -- compute score --
        grid_patches = search[:,:,burst_indices,nh_grid,:,:,:] 
        grid_patches = torch.cat([ref_patch,grid_patches],dim=2)
        score,count = 0,0
        for (nset0,nset1) in n_grids[:100]:

            # -- original --
            denoised0 = torch.mean(grid_patches[:,:,nset0],dim=2)
            denoised1 = torch.mean(grid_patches[:,:,nset1],dim=2)
            # score += F.mse_loss(denoised0,denoised1).item()

            # -- neurips 2019 --
            rep0 = repeat(denoised0,'r b c p1 p2 -> r b tile c p1 p2',tile=len(nset0))
            rep01 = repeat(denoised0,'r b c p1 p2 -> r b tile c p1 p2',tile=len(nset1))
            res0 = grid_patches[:,:,nset0] - rep0

            rep1 = repeat(denoised1,'r b c p1 p2 -> r b tile c p1 p2',tile=len(nset1))
            rep10 = repeat(denoised1,'r b c p1 p2 -> r b tile c p1 p2',tile=len(nset0))
            res1 = grid_patches[:,:,nset1] - rep1

            xterms01 = res0 + rep10
            xterms10 = res1 + rep01

            # score += F.mse_loss(xterms01,xterms10).item()
            score += F.mse_loss(xterms01,grid_patches[:,:,nset0]).item()
            score += F.mse_loss(xterms10,grid_patches[:,:,nset1]).item()

            count += 1
        score /= count

        # -- store best score --
        if score < best_score:
            best_score = score
            best_select = nh_grid

        # -- add score to results --
        scores[nh_index] = score

    if K == -1:
        return best_score,best_select
    else:
        indices = np.argsort(scores)[:K]
        scores_topK = scores[indices]
        indices_topK = nh_grids[indices]
        return scores_topK,indices_topK

def abp_search_pair(patches,burst_idx,PS,NH):
    R,B,N = patches.shape[:3]
    FMAX = np.finfo(np.float).max
    REF_NH = get_ref_nh(NH)
    ref_patch = patches[:,:,N//2,REF_NH,:,:,:]
    assert len(burst_idx) == 2, "Chunks of 2 right now."
    # best_value,best_select = FMAX*torch.ones(R,B),torch.zeros(R,B,2).type(torch.long)
    best_value,best_select = FMAX,None
    for i in range(NH**2):
        for j in range(NH**2):
            i_patch = patches[:,:,burst_idx[0],i,:,:,:] 
            j_patch = patches[:,:,burst_idx[1],j,:,:,:] 
            select = torch.LongTensor([i,j])
            ave_patches = torch.mean(patches[:,:,burst_idx,select,:,:,:],dim=2)

            value,count = 0,0
            compare_patches = [i_patch,j_patch,ave_patches,ref_patch]
            for k,k_patch in enumerate(compare_patches):
                for l,l_patch in enumerate(compare_patches):
                    if k >= l: continue
                    value += F.mse_loss(k_patch,l_patch).item()
                    # k_mean = k_patch.mean().item()
                    # l_mean = l_patch.mean().item()
                    # var = (75./255.)**2
                    # value += (F.mse_loss(k_patch-k_mean,l_patch-l_mean).item() - 2*var)**2
                    count += 1
            value /= count
            if value < best_value:
                best_value = value
                best_select = select
    return best_select

#
# ABP "Pairs" Search (old)
# 

def run_abp_patch_search_pairs_global_dynamics(patches,PS,NH):

    # -- shapes --
    R,B,N = patches.shape[:3]
    REF_NH = get_ref_nh(NH)
    mid_NH = torch.LongTensor([REF_NH])

    # -- init -- 
    best_indices = torch.zeros(B,N).type(torch.long)

    alpha = 4.
    pn_noisy = torch.poisson(alpha* (patches+0.5) )/alpha

    # -- run comparisons --
    for b in range(B):
        features_b = pn_noisy[:,[b]]

        if N == 5:
            left_idx = torch.LongTensor(np.r_[:N//2])
            best_left = abp_search_pair(features_b,left_idx,PS,NH)
    
            right_idx = torch.LongTensor(np.r_[N//2+1:N])
            best_right = abp_search_pair(features_b,right_idx,PS,NH)
        
            best_idx_b = torch.cat([best_left,mid_NH,best_right],dim=0).type(torch.long)
        elif N == 3:
            no_mid_idx = torch.LongTensor(np.r_[0,2])
            best_idx = abp_search_pair(features_b,no_mid_idx,PS,NH)
            best_idx_b = torch.cat([best_idx[[0]],mid_NH,best_idx[[1]]],dim=0).type(torch.long)
        best_indices[b] = best_idx_b
    return best_indices


#
# ABP Test Search
# 

def abp_test_search_old(cfg,burst):
    """
    burst: shape = (N,B,C,H,W)
    """

    # -- prepare variables --

    N,B,C,H,W = burst.shape
    n_grid = torch.arange(N)
    mid = torch.LongTensor([N//2])
    no_mid = torch.LongTensor(np.r_[np.r_[:N//2],np.r_[N//2+1:N]])
    PS = cfg.patchsize
    NH = cfg.nh_size

    # -- apply noise for testing --
    patches = tile_burst_patches(burst,PS,NH)
    gn_noisy = torch.normal(patches,75./255.)
    alpha = 4.
    pn_noisy = torch.poisson(alpha* (patches+0.5) )/alpha
    # patches.shape = (R,B,N,NH^2,C,PS,PS)

    
    unrolled_img = tile_batches_to_image(gn_noisy,NH,PS,H)
    tv_utils.save_image(unrolled_img,"gn_noisy.png",normalize=True,range=(-.5,.5))

    unrolled_img = tile_batches_to_image(pn_noisy,NH,PS,H)
    tv_utils.save_image(unrolled_img,"pn_noisy.png",normalize=True,range=(0.,1.))


    """
    For each N we pick 1 element from NH^2.

    Integer programming problem with state space size (NH^2)^(N-1)
    
    Dynamic programming to the rescue. Divide-and-conquer?
    """
    if N != 5: print("\n\n\n\n\n[WARNING]: This program is only coded for N == 5\n\n\n\n\n")

    best_idx = torch.zeros(B,N).type(torch.long)
    # features = gn_noisy
    features = pn_noisy
    # features = torch.zeros_like(pn_noisy)
    for b in range(B):
        features_b = features[:,[b]]
        REF_NH = get_ref_nh(NH)
        mid = torch.LongTensor([REF_NH])
        # m_patch = features_b[:,:,N//2,REF_NH,:,:,:]
        # s_features = features[:,:,no_mid,select,:,:,:] # (R,B,N-1,C,PS,PS)
        # ave_features = torch.mean(s_features,dim=2) # (R,B,C,PS,PS)
        # psnr = F.mse_loss(ave_features,m_patch).item()
    
        left_idx = torch.LongTensor(np.r_[:N//2])
        # best_left = torch.zeros(len(left_idx))
        best_left = abp_search_pair(features_b,left_idx,PS,NH)
    
        right_idx = torch.LongTensor(np.r_[N//2+1:N])
        # best_right = torch.zeros(len(right_idx))
        best_right = abp_search_pair(features_b,right_idx,PS,NH)
        
        best_idx_b = torch.cat([best_left,mid,best_right],dim=0).type(torch.long)
        best_idx[b] = best_idx_b

    best_idx = best_idx.to(patches.device)
    best_patches = []
    for b in range(B):
        patches_n = []
        for n in range(N):
            patches_n.append(patches[:,b,n,best_idx[b,n],:,:,:])
        patches_n = torch.stack(patches_n,dim=1)
        best_patches.append(patches_n)
    best_patches = torch.stack(best_patches,dim=1)
    # best_patches = torch.stack([patches[:,:,n,best_idx[b,n],:,:,:] for n in range(N)],dim=1)
    print("bp",best_patches.shape)
    ave_patches = torch.mean(best_patches,dim=2)
    set_imgs = rearrange(best_patches[:,:,:,:,PS//2,PS//2],'(h w) b n c -> b n c h w',h=H)
    ave_img = rearrange(ave_patches[:,:,:,PS//2,PS//2],'(h w) b c -> b c h w',h=H)
    print("A",ave_img.shape)
    return best_idx,ave_img,set_imgs
