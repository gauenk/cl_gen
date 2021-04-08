
# -- python imports --
import numpy as np
from einops import rearrange,repeat
from itertools import chain, combinations
from pathlib import Path
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils.misc import images_to_psnrs
from .tile_utils import *
from .abps_utils import *

def test_abp_global_search(cfg,clean,noisy_img=None):

    # -- create vars for search --
    N,B,C,H,W = clean.shape
    PS = cfg.patchsize
    NH = cfg.nh_size

    # -- setup for best_indices --
    patches = tile_burst_patches(clean,PS,NH)

    # -- add noise for testing --
    if noisy_img is None:
        alpha = 4.
        noisy = torch.poisson(alpha * (patches+0.5)) / alpha
    else: noisy = tile_burst_patches(noisy_img,PS,NH)

    # -- run search --
    scores,indices = test_run_abp_patch_search_exhaustive_global_dynamics(noisy,patches,PS,NH)
    indices.cuda(non_blocking=True)

    # -- recover aligned burst images --
    aligned = aligned_burst_image_from_indices_global_dynamics(patches,np.arange(N),indices)

    ave = torch.mean(aligned,dim=0)

    return indices,ave,aligned

def test_run_abp_patch_search_exhaustive_global_dynamics(noisy,clean,PS,NH):

    # -- shapes --
    R,B,N = noisy.shape[:3]
    REF_NH = get_ref_nh(NH)

    # -- init -- 
    best_scores = torch.zeros(B)
    best_indices = torch.zeros(B,N).type(torch.long)

    # -- run comparisons --
    for b in range(B):
        burst_indices = np.r_[np.r_[:N//2],np.r_[N//2+1:N]]
        args = [noisy[:,[b]],clean[:,[b]],burst_indices,PS,NH]
        scores_b,indices_b = test_abp_search_global_dynamics(*args)
        best_score_b,best_indices_b = scores_b[0],indices_b[0]
        best_indices_b = insert_nh_middle(best_indices_b,NH,burst_indices.shape[0])
        best_scores[b] = best_score_b
        best_indices[b] = best_indices_b
    return best_scores,best_indices

def test_abp_search_global_dynamics(patches,clean,burst_indices,PS,NH,K=2,verbose=False):
    
    if burst_indices.shape[0] <= 3:
        return test_abp_search_exhaustive_global_dynamics(patches,clean,burst_indices,PS,NH,K=K)

    # -- init shapes --
    R,B,N = patches.shape[:3]
    FMAX = np.finfo(np.float).max
    REF_NH = get_ref_nh(NH)
    BI = burst_indices.shape[0]
    H = int(np.sqrt(patches.shape[0]))
    
    # -- split the burst grid up --
    split_grids = create_split_burst_grid(N)
    S = split_grids.shape[0]

    # -- recurse --
    agg = edict()
    agg.scores = edict({str(n):[] for n in range(N)})
    agg.indices = edict({str(n):[] for n in range(N)})
    for s,burst_grid in enumerate(split_grids):
        scores,indices = test_abp_search_global_dynamics(patches,clean,burst_grid,PS,NH,K=K)
        
        # -- append to aggregate --
        for i,n_int in enumerate(burst_grid):
            n = str(n_int)
            agg.scores[n].extend(scores)
            agg.indices[n].extend(indices[:,i])

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
    print(top_nh_indices)
    top_nh_grid = create_grid_from_ndarrays(top_nh_indices)
    print(f"Top NH Grid {len(top_nh_grid)}")

    # -- final global search --
    scores,indices = test_abp_search_exhaustive_global_dynamics(patches,clean,
                                                                burst_indices,PS,NH,
                                                                K=K,nh_grids=top_nh_grid)

    

    return scores,indices

def test_abp_search_exhaustive_global_dynamics(noisy,clean,burst_indices,PS,NH,
                                               K=-1,nh_grids=None):

    # -- init vars --
    R,B,N = noisy.shape[:3]
    FMAX = np.finfo(np.float).max
    REF_NH = get_ref_nh(NH)
    print(f"REF_NH: {REF_NH}")
    BI = burst_indices.shape[0]
    ref_patch = noisy[:,:,[N//2],[REF_NH],:,:,:]

    # -- create clean testing image --
    H = int(np.sqrt(clean.shape[0]))
    clean_img = rearrange(clean[...,N//2,REF_NH,:,PS//2,PS//2],'(h w) b c -> b c h w',h=H)
    clean_img = repeat(clean_img,'b c h w -> tile b c h w',tile=N)

    # -- create search grids --
    if nh_grids is None: nh_grids = create_nh_grids(BI,NH)
    n_grids = create_n_grids(BI)
    print(f"NH_GRIDS {len(nh_grids)} | N_GRIDS {len(n_grids)}")

    # -- randomly initialize grids --
    # np.random.shuffle(nh_grids)
    # np.random.shuffle(n_grids)

    # -- init loop vars --
    psnrs = np.zeros( (len(nh_grids),BI))
    scores = np.zeros(len(nh_grids))
    scores_old = np.zeros(len(nh_grids))
    best_score,best_select = FMAX,None

    # -- remove boundary --
    aug_burst_indices = insert_n_middle( burst_indices, N )
    aug_burst_indices = torch.LongTensor(aug_burst_indices)
    subR = torch.arange(H*H//3*2)+NH*H
    search = noisy[subR]
    ref_patch = ref_patch[subR]

    # -- coordinate descent --
    for nh_index,nh_grid in enumerate(nh_grids):
        # -- compute score --
        grid_patches = search[:,:,burst_indices,nh_grid,:,:,:] 
        grid_patches = torch.cat([ref_patch,grid_patches],dim=2)
        score,score_old,count = 0,0,0
        for (nset0,nset1) in n_grids[:100]:
            denoised0 = torch.mean(grid_patches[:,:,nset0],dim=2)
            denoised1 = torch.mean(grid_patches[:,:,nset1],dim=2)
            score_old += F.mse_loss(denoised0,denoised1).item()
            

            # -- neurips 2019 --
            rep0 = repeat(denoised0,'r b c p1 p2 -> r b tile c p1 p2',tile=len(nset0))
            rep01 = repeat(denoised0,'r b c p1 p2 -> r b tile c p1 p2',tile=len(nset1))
            res0 = grid_patches[:,:,nset0] - rep0

            rep1 = repeat(denoised1,'r b c p1 p2 -> r b tile c p1 p2',tile=len(nset1))
            rep10 = repeat(denoised1,'r b c p1 p2 -> r b tile c p1 p2',tile=len(nset0))
            res1 = grid_patches[:,:,nset1] - rep1

            n0,n1 = len(nset0),len(nset1)
            xterms0,xterms1 = np.mgrid[:n0,:n1]
            xterms0,xterms1 = xterms0.ravel(),xterms1.ravel()
            # print(xterms0.shape,xterms1.shape,res0.shape,xterms0.max(),xterms1.max())
            score += F.mse_loss(res0[:,:,xterms0],res1[:,:,xterms1]).item()

            # xterms01 = res0 + rep10
            # xterms10 = res1 + rep01

            # score += F.mse_loss(xterms01,xterms10).item()
            # score += F.mse_loss(xterms01,grid_patches[:,:,nset0]).item()
            # score += F.mse_loss(xterms10,grid_patches[:,:,nset1]).item()

            count += 1
        score /= count

        # -- store best score --
        if score < best_score:
            best_score = score
            best_select = nh_grid

        # -- add score to results --
        scores[nh_index] = score
        scores_old[nh_index] = score_old

        # -- compute and store psnrs --
        pgrid = insert_nh_middle( nh_grid, NH, BI )[None,]
        bgrid = aug_burst_indices
        nh_grid = nh_grid[None,]
        rec_img = aligned_burst_image_from_indices_global_dynamics(clean,burst_indices,nh_grid)#bgrid,pgrid)
        nh_psnrs = images_to_psnrs(rec_img,clean_img[burst_indices])
        psnrs[nh_index,:] = nh_psnrs


    score_idx = np.argmin(scores)
    print(f"Best Score [{scores[score_idx]}] PSNRS @ [{score_idx}]:",psnrs[score_idx])

    psnr_idx = np.argmax(np.mean(psnrs,1))
    print(f"Best PSNR @ [{psnr_idx}]",psnrs[psnr_idx])
    # print(scores[score_idx] - scores[psnr_idx])

    old_score_idx = np.argmin(scores_old)
    print(f"Best OLD Score [{scores_old[old_score_idx]}] PSNRS @ [{old_score_idx}]:",psnrs[old_score_idx])
    print(f"Current Score @ OLD Score [{scores[old_score_idx]}]")
    print(f"[Old score idx v.s. Current score idx v.s. Best PSNR] {old_score_idx} v.s. {score_idx} v.s. {psnr_idx}")


    #
    #  Recording Score Info
    #

    # -- save score info --
    scores /= np.sum(scores)
    score_fn = f"scores_{NH}_{N}_{len(nh_grids)}_{len(n_grids)}"
    txt_fn = Path(f"output/abps/{score_fn}.txt")
    np.savetxt(txt_fn,scores)

    # -- plot score --
    plot_fn = Path(f"output/abps/{score_fn}.png")
    fig,ax = plt.subplots(figsize=(8,8))
    ax.plot(np.arange(scores.shape[0]),scores,'-+')
    ax.axvline(x=psnr_idx,color="r")
    ax.axvline(x=score_idx,color="k")
    plt.savefig(plot_fn,dpi=300)
    plt.close("all")


    #
    #  Recording PSNR Info
    #

    # -- save score info --
    psnr_fn = f"psnrs_{NH}_{N}_{len(nh_grids)}_{len(n_grids)}"
    txt_fn = Path(f"output/abps/{psnr_fn}.txt")
    np.savetxt(txt_fn,psnrs)

    # -- plot psnr --
    plot_fn = Path(f"output/abps/{psnr_fn}.png")
    fig,ax = plt.subplots(figsize=(8,8))
    ax.plot(np.arange(psnrs.shape[0]),psnrs,'-+')
    ax.axvline(x=psnr_idx,color="r")
    ax.axvline(x=score_idx,color="k")
    plt.savefig(plot_fn,dpi=300)
    plt.close("all")

    print(f"Wrote {score_fn} and {psnr_fn}")

    if K == -1:
        return best_score,best_select
    else:
        search_indices_topK = np.argsort(scores)[:K]
        scores_topK = scores[search_indices_topK]
        nh_grids_topK = nh_grids[search_indices_topK]
        return scores_topK,nh_grids_topK



