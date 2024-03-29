
# -- python imports --
import math
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from itertools import chain, combinations

# -- numba imports --
import numba
from numba import jit,prange,cuda

# -- pytorch imports --
import torch
import torch.nn.functional as F

# -- project imports --
from pyutils import torch_xcorr,create_combination,print_tensor_stats,save_image,create_subset_grids,create_subset_grids,create_subset_grids_fixed,ncr,sample_subset_grids
from layers.unet import UNet_n2n,UNet_small
from layers.ot_pytorch import sink_stabilized,pairwise_distances,dmat

# -- [local] project imports --
from ..utils import get_ref_block_index
from .bootstrap_numba import compute_bootstrap,fill_weights,fill_weights_pix
from ._indexing import index_along_frames

def vprint(*args):
    # vprint_emacs_search
    verbose = False
    if verbose:
        print(*args)

def get_score_functions(names):
    scores = edict()
    for name in names:
        scores[name] = get_score_function(name)
    return scores

def get_score_function(name):
    if name == "ave":
        return ave_score
    elif name == "mse":
        return mse_score
    elif name == "gaussian_ot":
        return gaussian_ot_score
    elif name == "emd":
        return emd_score        
    elif name == "powerset":
        return powerset_score
    elif name == "extrema":
        return extrema_score
    elif name == "smsubset":
        return smsubset_score
    elif name == "lgsubset":
        return lgsubset_score
    elif name == "lgsubset_v_indices":
        return lgsubset_v_indices_score
    elif name == "lgsubset_v_ref":
        return lgsubset_v_ref_score
    elif name == "powerset_v_indices":
        return powerset_v_indices_score
    elif name == "powerset_v_ref_score":
        return powerset_v_ref_score
    elif name == "pairwise":
        return pairwise_delta_score
    elif name == "refcmp":
        return refcmp_score
    elif name == "jackknife":
        return jackknife
    elif name == "bootstrapping":
        return bootstrapping
    elif name == "bootstrapping_mod1":
        return bootstrapping_mod1
    elif name == "bootstrapping_mod2":
        return bootstrapping_mod2
    elif name == "bootstrapping_mod3":
        return bootstrapping_mod3
    elif name == "bootstrapping_mod4":
        return bootstrapping_mod4
    elif name == "bootstrapping_mod5":
        return bootstrapping_mod5
    elif name == "bootstrapping_mod6":
        return bootstrapping_mod6
    elif name == "bootstrapping_limitB":
        return bootstrapping_limitB
    elif name == "bootstrapping_cf":
        return bootstrapping_cf
    elif name == "sim_trm":
        return sim_trm
    elif name == "ransac":
        return ransac
    elif name == "shapley":
        return shapley
    else:
        raise ValueError(f"Uknown score function [{name}]")

def bootstrapping_cf(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    nframes = T
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) b e')
    samples = samples.contiguous() # speed up?

    # -- compute closed form (cf) bootstrap --
    term_1 = torch.sum(samples**2,dim=0)/nframes**2
    term_2 = torch.mean(samples,dim=0)**2/nframes
    boot_cf = torch.mean(term_1 - term_2,dim=0)


    # -- add back patchsize for compat --
    scores = boot_cf
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores,'r b e -> r b e t',t=T)

    return scores,scores_t

    
def bootstrapping(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    nframes = T
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2
    ave = torch.mean(samples,dim=0)

    # -- compute ave diff between model and subsets --
    scores = torch.zeros(B*E,device=device)
    scores_t = torch.zeros(T,B*E,device=device)
    counts_t = torch.zeros(T,1,device=device)
    # mask = torch.zeros(T,1,device=device)
    nbatches,batch_size = 3,1000
    subsets = torch.LongTensor(sample_subset_grids(nbatches*batch_size,T))
    subsets = rearrange(subsets,'(nb bs) t -> nb bs t',nb=nbatches)
    # vprint("subsets.shape ",subsets.shape)
    # compute_bootstrap(samples,scores_t,counts_t,ave,subsets,nbatches,batch_size)

    for batch_idx in range(nbatches):
        # subsets = torch.LongTensor(sample_subset_grids(batch_size,T))
        for subset in subsets[batch_idx]:
            # mask[...] = 0
            # mask[subset] = 1
            counts_t[subset] += 1
            subset_pix = samples[subset]
            vprint("batch_idx",batch_idx)
            vprint("subsets.shape",subsets.shape)
            vprint("subsets[batch_idx].shape",subsets[batch_idx].shape)
            vprint("subset.shape",subset.shape)
            vprint("subset_pix.shape",subset_pix.shape)
            subset_ave = torch.mean(subset_pix,dim=0)
            # vprint("subset_ave.shape",subset_ave.shape)
            loss = torch.mean( (subset_ave - ave)**2, dim=0)
            # vprint("loss.shape",loss.shape)
            scores_t[subset] += loss
            scores += loss/(nbatches*batch_size)

    scores_t /= counts_t
    # scores = torch.mean(scores_t,dim=0)
    scores_t = scores_t.T # (T,E) -> (E,T)
    vprint("scores.shape",scores.shape)

    # -- no cuda --
    scores = rearrange(scores.cpu(),'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t.cpu(),'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    e = scores.shape[-1]
    if True:#e == 1:
        print("--> bootstrap original <--")
        print("scores.shape ",scores.shape)
        print("samples.shape ",samples.shape)
        print(samples[:,30:40,10])
        mse_scores,mse_scores_t = mse_score(cfg,expanded)
        bs2_scores,bs2_scores_t = bootstrapping_mod2(cfg,expanded)
        print("mse_scores.shape ",mse_scores.shape)
        print("mse_scores_t.shape ",mse_scores_t.shape)
        prop = scores[0,30:33].cpu().numpy()
        gt = mse_scores[0,30:33].cpu().numpy()
        bs2 = bs2_scores[0,30:33].cpu().numpy()
        print("prop",prop)
        print("mse",gt)
        print("bs2",bs2)
        print("abs [prop-gt]",np.abs(prop - gt))
        print("abs [prop-bs2]",np.abs(prop - bs2))
        print("abs [bs2-gt]",np.abs(bs2 - gt))
        print("abs_nmlz [prop,gt]",np.abs(prop - gt)/gt)
        print("abs_nmlz [bs2,prop]",np.abs(prop - bs2)/prop)
        print("abs_nmlz [bs2,gt]",np.abs(bs2 - gt)/gt)
        print("ratio: gt/prop",gt/prop)
        print("ratio: gt/bs2",gt/bs2)
        print("ratio: prop/bs2",prop/bs2)


    return scores,scores_t

def bootstrapping_mod1(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    nframes = T
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2
    # ave = torch.mean(samples,dim=0)

    # -- compute ave diff between model and subsets --
    scores_t = torch.zeros(T,B*E,device=device)
    weights = torch.zeros(T,1,1,device=device)
    counts_t = torch.zeros(T,1,device=device)
    nbatches,batch_size = 10,500
    subsets = torch.LongTensor(sample_subset_grids(nbatches*batch_size,T))
    subsets = rearrange(subsets,'(nb bs) t -> nb bs t',nb=nbatches)
    # compute_bootstrap(samples,scores_t,counts_t,ave,subsets,nbatches,batch_size)

    for batch_idx in range(nbatches):
        # subsets = torch.LongTensor(sample_subset_grids(batch_size,T))
        for subset in subsets[batch_idx]:
            subsize = len(subset)
            counts_t[subset] += 1
            weights[...] = -1./nframes
            weights[subset] += 1./subsize
            loss = torch.mean(torch.sum(weights * samples,dim=0),dim=0)
            scores_t[subset] += loss

    scores_t /= counts_t
    scores = torch.mean(scores_t,dim=0)
    scores_t = scores_t.T # (T,E) -> (E,T)
    vprint("scores.shape",scores.shape)

    # -- no cuda --
    scores = rearrange(scores.cpu(),'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t.cpu(),'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def bootstrapping_mod2(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    nframes = T
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> (r c h w) t (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2
    # nbatches,batchsize = 80,25
    nbatches,batchsize = 50,500
    nsubsets = nbatches*batchsize

    scores = torch.zeros(B*E,device=device)
    scores_t = torch.zeros(B*E,T,device=device)
    counts = torch.zeros((batchsize,nframes),device=device)
    weights = torch.zeros((batchsize,nframes),device=device)

    # -- cuda memory --
    # t = torch.cuda.get_device_properties(cfg.gpuid).total_memory
    # r = torch.cuda.memory_reserved(cfg.gpuid)
    # a = torch.cuda.memory_allocated(cfg.gpuid)
    # f = r-a  # free inside reserved
    # print(samples.shape)
    # print(f"gpu:{cfg.gpuid}",t,f)

    for batch_idx in range(nbatches):
        weights,counts = fill_weights(weights,counts,batchsize,nframes,cfg.gpuid)
        counts_t = torch.sum(counts,dim=0)[:,None]
        wsamples = weights @ samples
        wsamples2 = torch.pow(wsamples,2)
        # print("wsamples2.shape ",wsamples2.shape)
        w_pix_ave = torch.mean(wsamples2,dim=0)
        w_pix_ave_nmlz = (counts.T @ w_pix_ave) / counts_t
        scores_t += w_pix_ave_nmlz.T/nbatches
        # scores += torch.mean(w_pix_ave_nmlz,dim=0)
        scores += torch.mean(w_pix_ave,dim=0)/nbatches

        # -- cuda memory --
        # t = torch.cuda.get_device_properties(cfg.gpuid).total_memory
        # r = torch.cuda.memory_reserved(cfg.gpuid)
        # a = torch.cuda.memory_allocated(cfg.gpuid)
        # f = r-a  # free inside reserved
        # print(f"gpu:{cfg.gpuid}",t,f)
    
        # torch.cuda.empty_cache()

    scores = rearrange(scores,'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t,'(b e) t -> b e t',b=B)

    # -- add back num of "same-motion" patches for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)
    e = scores.shape[2]
    # if e == 1:
    #     print("scores.shape ",scores.shape)
    #     print("samples.shape ",samples.shape)
    #     mse_scores,mse_scores_t = mse_score(cfg,expanded)
    #     print("mse_scores.shape ",mse_scores.shape)
    #     print("mse_scores_t.shape ",mse_scores_t.shape)
    #     prop = scores[0,:3].cpu().numpy()
    #     gt = mse_scores[0,:3].cpu().numpy()
    #     print(prop)
    #     print(gt)
    #     print(np.abs(prop - gt))
    #     print(np.abs(prop - gt)/gt)
    #     print(gt/prop)

    return scores,scores_t

def bootstrapping_limitB(cfg,patches,prev_patches,prev_scores,dframes,patches_full):
    if cfg.bs_type == "full":
        return bootstrapping_mod2(cfg,patches,prev_patches,prev_scores,dframes,patches_full)
    elif cfg.bs_type == "step":
        return bootstrapping_limitB_impl(cfg,patches,prev_patches,prev_scores,dframes,patches_full)
    else:
        raise ValueError(f"Uknown bootstrapping type {cfg.bs_type}")

def bootstrapping_limitB_impl(cfg,patches,prev_patches,prev_scores,dframes,patches_full):

    # ------------------------
    #   Setup Function Call
    # ------------------------

    # -- hyperparams --
    nbatches,batchsize = 10,25
    nsubsets = nbatches*batchsize
    nframes = cfg.nframes

    # -- reshaping --
    # patches = rearrange(patches,'r b e c h w -> e b (r c h w)')
    # patches = patches.contiguous() # speed up?
    nimages,npatches,naligns,nftrs = patches.shape
    # prev_patches = rearrange(prev_patches,'r b e c h w -> e b (r c h w)')
    nimages,npatches,naligns_prev,nframes,nftrs = prev_patches.shape    
    assert naligns_prev == 1, "Only one previous alignment is supported."
    nimages,npatches,naligns_prev = prev_scores.shape
    assert naligns_prev == 1, "Only one previous alignment is supported."

    # -- create empty vars --
    device = patches.device
    dframes = dframes.to(device)
    print("dframes.shape ",dframes.shape)
    # prev_patches = prev_patches.to(device)
    # patches = patches.to(device)
    scores = torch.zeros(nimages,npatches,naligns,device=device)
    scores_t = torch.zeros(nimages,npatches,naligns,nframes,device=device)

    # --------------------------
    #   Compute Delta(i,j,j') 
    # --------------------------
    # [shape (npatches,naligns)]

    # -- average of prev patches RM dframes --
    # consts = torch.mean(prev_patches,dim=3) # average over specificy frame alignment
    print("-"*10 + " patches.")
    pix_idx = 30
    frames = dframes[0,pix_idx]
    print("patches_full.shape ",patches_full.shape)
    print(patches_full[0,pix_idx,:,:,:2])
    print(prev_patches[0,pix_idx,0,:,:2])
    print(patches_full[0,pix_idx,:,:,:2].shape)
    print(prev_patches[0,pix_idx,0,:,:2].shape)
    print(dframes[0,pix_idx])
    print("-"*10 + " means.")
    consts = compute_mean_rm_dframes(prev_patches,dframes)
    consts_other = compute_mean_rm_dframes_other(patches_full,dframes)
    print("Diff should be zero: ",torch.sum(torch.abs(consts - consts_other)))
    print(consts.shape)
    print(consts[0,0,:,0])
    print(consts_other[0,0,:,0])
    print(consts[0,30,:,0])
    print(consts_other[0,30,:,0])
    print("patches_full.shape ",patches_full.shape)

    # -- constants --
    sq_coeff = 1./(nframes**2.) - 1./(nframes**3.)
    lin_coeff = -2./nframes**3

    # -- squared term --
    j = [0]
    sq_diff = torch.zeros(nimages,npatches,naligns,nftrs,device=device)
    print("naligns ",naligns)
    for p in range(npatches):
        for a in range(naligns):
            dframe = dframes[0,p,a]
            # sq_diff[0,p,a] = patches[0,p,a]**2 - prev_patches[0,p,j[0],dframe]**2
            # if p == 30:
            #     print(patches_full[0,p,a,dframe]**2,prev_patches[0,p,j[0],dframe]**2)
            term1 = patches_full[0,p,a,dframe]**2
            term2 = prev_patches[0,p,j[0],dframe]**2
            sq_diff[0,p,a] = term1 - term2
            
    sq_diff = torch.mean(sq_diff,dim=-1)
    print("sq_diff ",sq_diff[0,0].cpu().numpy())
    sq_term = sq_coeff * sq_diff
    print("sq_term @ 0 ",sq_term[0,0].cpu().numpy())
    print("sq_term @ 20 ",sq_term[0,20].cpu().numpy())
    print("sq_term @ 30 ",sq_term[0,30].cpu().numpy())

    # -- linear term --
    lin_diff = torch.zeros(nimages,npatches,naligns,nftrs,device=device)
    print("pre lin.")
    print(consts[0,0,:,0])
    print(dframes[0,0,:])
    print(patches[0,0,:,0])
    print(prev_patches[0,0,j[0],:,0])
    # print(torch.sum(torch.abs(patches[0,0,:,:] - prev_patches[0,0,j[0],:,:])))
    for p in range(npatches):
        for a in range(naligns):
            const = consts[0,p,a]
            dframe = dframes[0,p,a]
            #lin_diff[0,p,a] = const * (patches[0,p,a] - prev_patches[0,p,j[0],dframe])
            lin_diff[0,p,a] = const * (patches_full[0,p,a,dframe] - prev_patches[0,p,j[0],dframe])
    lin_diff = torch.mean(lin_diff,dim=-1)
    print("lin_diff ",lin_diff[0,0].cpu().numpy())
    lin_term = lin_coeff * lin_diff
    print("lin_term @ 0 ",lin_term[0,0].cpu().numpy())
    print("lin_term @ 20 ",lin_term[0,20].cpu().numpy())
    print("lin_term @ 30 ",lin_term[0,30].cpu().numpy())

    Delta = sq_term + lin_term
    print("Delta.shape ",Delta.shape)
    print(Delta[0,1,:])
    print(scores.shape)
    print(prev_scores.shape)
    # npatches,naligns,naligns_prev = Delta.shape

    # ----------------------------------------------
    #   Compute BS(i,j') = BS(i,j) + Delta(i,j,j')
    # ----------------------------------------------
    # [shape (npatches,naligns)]

    scores = prev_scores + Delta
    print("scores.shape ",scores.shape)

    return scores,scores_t
    

def compute_mean_rm_dframes_other(patches,dframes):

    # -- shapes and alloc --
    device = patches.device
    print(patches.shape)
    nimages,npatches,naligns = dframes.shape
    nimages,npatches,naligns,nframes,nftrs = patches.shape
    psd = torch.zeros(nimages,npatches,naligns,nframes-1,nftrs,device=device)

    # -- to numba tensors --
    patches_nba = cuda.as_cuda_array(patches)
    psd_nba = cuda.as_cuda_array(psd)
    dframes_nba = cuda.as_cuda_array(dframes)

    # -- cuda kernel stats --
    threads_per_block = (32,32)
    tpb = threads_per_block
    blocks_patches = npatches//tpb[0] + (npatches%tpb[0]!=0)
    blocks_aligns = naligns//tpb[1] + (naligns%tpb[1]!=0)
    blocks = (blocks_patches,blocks_aligns)

    # -- launch each batch separately --
    for i in range(nimages):
        args = (psd_nba[i],patches_nba[i],dframes_nba[i])
        index_along_frames_cuda_other[blocks,threads_per_block](*args)
    aves = torch.sum(psd,dim=3)
    return aves

def compute_mean_rm_dframes(patches,dframes):

    # -- shapes and alloc --
    device = patches.device
    nimages,npatches,naligns = dframes.shape
    nimages,npatches,naligns_prev,nframes,nftrs = patches.shape
    assert naligns_prev == 1,"Must be one."
    patches = patches[:,:,0]
    psd = torch.zeros(nimages,npatches,naligns,nframes-1,nftrs,device=device)

    # -- to numba tensors --
    patches_nba = cuda.as_cuda_array(patches)
    psd_nba = cuda.as_cuda_array(psd)
    dframes_nba = cuda.as_cuda_array(dframes)
    # print(type(patches_nba))
    # print(type(psd_nba))
    # print(type(dframes_nba))
    # print(psd.__cuda_array_interface__)
    # print(psd_nba.__cuda_array_interface__)
    # print(dframes.__cuda_array_interface__)
    # print(dframes_nba.__cuda_array_interface__)


    # -- cuda kernel stats --
    threads_per_block = (32,32)
    tpb = threads_per_block
    blocks_patches = npatches//tpb[0] + (npatches%tpb[0]!=0)
    blocks_aligns = naligns//tpb[1] + (naligns%tpb[1]!=0)
    blocks = (blocks_patches,blocks_aligns)

    # -- launch each batch separately --
    for i in range(nimages):
        args = (psd_nba[i],patches_nba[i],dframes_nba[i])
        index_along_frames_cuda[blocks,threads_per_block](*args)
        # args = (psd[i],patches[i],dframes[i])
        # index_along_frames(*args)
        
    # pix_idx = 16*8+4
    # print(psd[0,pix_idx,:,:,:2].shape)
    # print(psd[0,pix_idx,:,:,:2])
    # print(patches[0,pix_idx,:,:2].shape,patches.shape)
    # print(patches[0,pix_idx,:,:2])
    # print(dframes[0,pix_idx])
    print("aves rm dframe")
    print("-"*10)
    print(dframes[0,0])
    print(psd[0,0,:,:,0])
    print(psd_nba[0,0,:,:,0])
    print(torch.as_tensor(psd_nba)[0,0,:,:,0])
    print(patches[0,0,:,0])
    print(patches[0,0,:,:2])

    print("-"*10)
    print(dframes[0,20])
    print(psd[0,20,:,:,0])
    print(torch.as_tensor(psd_nba)[0,20,:,:,0])
    print("-"*10)
    print(patches[0,20,:,0])
    print(patches[0,20,:,:5])
    print(patches[0,21,:,:5])
    print(patches[0,19,:,:5])

    aves = torch.sum(psd,dim=3)

    return aves

def index_along_frames(psd,patches,dframes):

    npatches,naligns,nframes_m1,nftrs = psd.shape
    nframes = nframes_m1 + 1
    for p_idx in range(npatches):
        for a_idx in range(naligns):
            t_idx = 0
            for t in range(nframes):
                if t == dframes[p_idx,a_idx]: continue
                for f in range(nftrs):
                    psd[p_idx,a_idx,t_idx,f] = patches[p_idx,t,f]
                t_idx += 1
        
@cuda.jit
def index_along_frames_cuda_other(psd,patches,dframes):
    p_idx,a_idx = cuda.grid(2)
    npatches,naligns,nframes_m1,nftrs = psd.shape
    nframes = nframes_m1 + 1
    if p_idx < npatches and a_idx < naligns:
        t_idx = 0
        for t in range(nframes):
            if t == dframes[p_idx,a_idx]: continue
            # if p_idx == 20: print(p_idx,a_idx,t_idx,t)
            for f in range(nftrs):
                # if p_idx == 20: print(patches[p_idx,0,t,f])
                psd[p_idx,a_idx,t_idx,f] = patches[p_idx,a_idx,t,f]
            t_idx += 1
            # assert t_idx < nframes, "must be less than nframes."

@cuda.jit
def index_along_frames_cuda(psd,patches,dframes):
    p_idx,a_idx = cuda.grid(2)
    npatches,naligns,nframes_m1,nftrs = psd.shape
    nframes = nframes_m1 + 1
    # if p_idx > npatches or a_idx > naligns: return
    if p_idx < npatches and a_idx < naligns:
        t_idx = 0
        for t in range(nframes):
            if t == dframes[p_idx,a_idx]: continue
            # if p_idx == 20: print(p_idx,a_idx,t_idx,t)
            for f in range(nftrs):
                # if p_idx == 20: print(patches[p_idx,0,t,f])
                psd[p_idx,a_idx,t_idx,f] = patches[p_idx,t,f]
            t_idx += 1
            # assert t_idx < nframes, "must be less than nframes."
    
def bootstrapping_mod3(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    nframes = T
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> c (r h w) t (b e)')
    samples = samples.contiguous() # speed up?
    ncolor,npix,nframes,nsamples = samples.shape
    t_ref = nframes//2
    nbatches,batchsize = 20,250
    nsubsets = nbatches*batchsize

    # -- init zeros --
    scores = torch.zeros(nsamples,device=device)
    scores_t = torch.zeros(nsamples,nframes,device=device)
    counts = torch.zeros((batchsize,npix,nframes),device=device)
    weights = torch.zeros((batchsize,npix,nframes),device=device)

    # -- run batches --
    for batch_idx in range(nbatches):
        weights,counts = fill_weights_pix(weights,counts,batchsize,
                                          npix,nframes,cfg.gpuid)
        # print(weights[0,0,:3])
        # counts_t = torch.sum(counts,dim=0)[:,None]

        wsamples = weights.transpose(1,0) @ samples
        wsamples = torch.pow(wsamples,2)
        wsamples = torch.mean(wsamples,dim=(0,1))
        # print("wsamples.shape ",wsamples.shape)
        # print("counts.shape ",counts.shape)
        
        # wsamples_nmlz = (counts.T @ wsamples) / counts_t
        # scores_t += wsamples_nmlz.T
        # scores += torch.mean(wsamples_nmlz,dim=0)
        scores += torch.mean(wsamples,dim=0)
    scores_t /= nbatches
    scores /= nbatches

    # -- reshape --
    scores = rearrange(scores,'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t,'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def bootstrapping_mod4(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    nframes = T
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> (r c h w) t (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2
    nbatches,batchsize = 10,500
    nsubsets = nbatches*batchsize

    scores = torch.zeros(B*E,device=device)
    scores_t = torch.zeros(B*E,T,device=device)
    counts = torch.zeros((batchsize,nframes),device=device)
    weights = torch.zeros((batchsize,nframes),device=device)
    for batch_idx in range(nbatches):
        weights,counts = fill_weights(weights,counts,batchsize,nframes,cfg.gpuid)
        counts_t = torch.sum(counts,dim=0)[:,None]
        # print("weights.shape ",weights.shape)
        # print("samples.shape ",samples.shape)
        # print("counts_t.shape ",counts_t.shape)
        wsamples = weights @ samples
        wsamples2 = torch.pow(wsamples,2)
        # print("wsamples2.shape ",wsamples2.shape)
        w_pix_ave = torch.mean(wsamples2,dim=0)
        # print("w_pix_ave.shape ",w_pix_ave.shape)
        w_pix_ave_nmlz = (counts.T @ w_pix_ave) / counts_t
        # print("w_pix_ave_nmlz.shape ",w_pix_ave_nmlz.shape)
        scores_t += w_pix_ave_nmlz.T
        # print("scores_t.shape: ",scores_t.shape)
        # scores += torch.mean(w_pix_ave_nmlz,dim=0)
        # scores += torch.mean(w_pix_ave,dim=0)
        delta = torch.abs(w_pix_ave[:-1] - w_pix_ave[1:])
        scores += torch.mean(delta,dim=0)
    scores_t /= nbatches
    scores /= nbatches

    scores = rearrange(scores,'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t,'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def bootstrapping_mod5(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    nframes = T
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> (r c h w) t (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2
    nbatches,batchsize = 10,500
    nsubsets = nbatches*batchsize

    scores = torch.zeros(B*E,device=device)
    scores_t = torch.zeros(B*E,T,device=device)
    counts = torch.zeros((batchsize,nframes),device=device)
    weights = torch.zeros((batchsize,nframes),device=device)
    for batch_idx in range(nbatches):
        weights,counts = fill_weights(weights,counts,batchsize,nframes,cfg.gpuid)
        counts_t = torch.sum(counts,dim=0)[:,None]
        # print("weights.shape ",weights.shape)
        # print("samples.shape ",samples.shape)
        # print("counts_t.shape ",counts_t.shape)
        wsamples = weights @ samples
        wsamples2 = torch.pow(wsamples,2)
        # print("wsamples2.shape ",wsamples2.shape)
        w_pix_ave = torch.mean(wsamples2,dim=0)
        # print("w_pix_ave.shape ",w_pix_ave.shape)
        w_pix_ave_nmlz = (counts.T @ w_pix_ave) / counts_t
        # print("w_pix_ave_nmlz.shape ",w_pix_ave_nmlz.shape)
        scores_t += w_pix_ave_nmlz.T
        # print("scores_t.shape: ",scores_t.shape)
        scores += torch.mean(w_pix_ave_nmlz,dim=0)
        # scores += torch.mean(w_pix_ave,dim=0)
        # delta = torch.abs(w_pix_ave[:-1] - w_pix_ave[1:])
        # scores += torch.mean(delta,dim=0)
    scores_t /= nbatches
    scores /= nbatches

    scores = rearrange(scores,'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t,'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def bootstrapping_mod6(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    nframes = T
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> (r c h w) t (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2
    ref_t = T//2
    nbatches,batchsize = 10,500
    nsubsets = nbatches*batchsize

    scores = torch.zeros(B*E,device=device)
    scores_t = torch.zeros(B*E,T,device=device)
    counts = torch.zeros((batchsize,nframes),device=device)
    weights = torch.zeros((batchsize,nframes),device=device)
    for batch_idx in range(nbatches):
        weights,counts = fill_weights(weights,counts,batchsize,nframes,cfg.gpuid)
        counts_t = torch.sum(counts,dim=0)[:,None]
        # print("weights.shape ",weights.shape)
        # print("samples.shape ",samples.shape)
        # print("counts_t.shape ",counts_t.shape)
        wsamples = weights @ samples
        wsamples2 = torch.pow(wsamples,2)
        # print("wsamples2.shape ",wsamples2.shape)
        w_pix_ave = torch.mean(wsamples2,dim=0)
        # print("w_pix_ave.shape ",w_pix_ave.shape)
        w_pix_ave_nmlz = (counts.T @ w_pix_ave) / counts_t
        # print("w_pix_ave_nmlz.shape ",w_pix_ave_nmlz.shape)
        scores_t += w_pix_ave_nmlz.T
        # print("scores_t.shape: ",scores_t.shape)
        # scores += torch.mean(w_pix_ave_nmlz,dim=0)
        # scores += torch.mean(w_pix_ave,dim=0)
        delta_l = torch.mean(torch.abs(w_pix_ave[ref_t] - w_pix_ave[0:ref_t]),dim=0)
        delta_r = torch.mean(torch.abs(w_pix_ave[ref_t] - w_pix_ave[ref_t+1:]),dim=0)
        scores += delta_l + delta_r #torch.mean(delta,dim=0)
    scores_t /= nbatches
    scores /= nbatches

    scores = rearrange(scores,'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t,'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def shapley(cfg,expanded):
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    # -- setup --
    R,B,E,T,C,H,W = expanded.shape
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2

    def compute_pair_ave(x,y):
        mom_1 = torch.mean((x - y)**2,dim=0)
        return mom_1

    def convert_subsets(subsets,t):
        filtered_subsets = []
        for subset in subsets:
            if t in subset:
                filtered_subsets.append(subset)
        return filtered_subsets

    def join_subset(subset,t):
        return np.r_[subset,[t]]

    # -- create subset grids --
    minSN,maxSN,max_num = 1,15,100000
    indices = np.arange(T)
    subsets = create_subset_grids(minSN,maxSN,indices,max_num)
    size = 15 # 12 got (a) win and (b) match

    # -- init loop --
    ave = torch.mean(samples,dim=0)
    scores_t = torch.zeros(B*E,T,device=device)
    for t in range(T):
        subsets_rm_t = convert_subsets(subsets,t)
        for subset_rm_t in subsets_rm_t:

            # -- create new subset --
            subset_with_t = join_subset(subset_rm_t,t)
            
            # -- grab images --
            ave_with_t = torch.mean(samples[subset_with_t],dim=0)
            ave_rm_t = torch.mean(samples[subset_rm_t],dim=0)

            # -- compute differences --
            v_with_t = compute_pair_ave(ave_with_t,ave)
            v_rm_t = compute_pair_ave(ave_rm_t,ave)
            v_diff = -(v_with_t - v_rm_t)

            # -- compute normalization --
            s = len(subsets_rm_t)
            Z = ncr(s,T)**(-1)

            # -- accumulate --
            scores_t[:,t] += Z * v_diff

    # -- compute mean over frames --
    scores = torch.mean(scores_t,dim=1)

    # -- no cuda --
    scores = rearrange(scores.cpu(),'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t.cpu(),'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def ransac(cfg,expanded):
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    # print("-> Print Time <-","\n\n\n")
    R,B,E,T,C,H,W = expanded.shape
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2

    
    def compute_std(gt_std,n_tr,n_te):
        std = np.sqrt(gt_std**2/n_tr + gt_std**2/n_te)
        return std

    def compute_pair_ave(x,y):
        mom_1 = torch.mean(torch.abs(x - y),dim=0)
        # mom_1 = torch.mean(x - y,dim=0)**2
        return mom_1

    def compute_pair_mom(x,y,nx,ny):
        mom_1 = (torch.mean(x,dim=0) - torch.mean(y,dim=0))**2
        mom_2 = (ny*torch.std(x,dim=0)**2 - nx*torch.std(y,dim=0)**2)**2
        return mom_1 + mom_2

    def compute_gaussian_ot(dist,gt_std):
        ave = torch.mean(dist,dim=0)
        std = torch.std(dist,dim=0)
        ot_loss = ave**2 + (std**2 - gt_std**2)**2
        return ot_loss

    def compute_loss_oos(tr_points,te_points,gt_std):
        # est.shape, N BE D
        # tr_ave = compute_train_est(tr_points)
        n_tr,n_te = len(tr_points),len(te_points)
        tr_ave = torch.mean(tr_points,dim=0)                
        te_ave = torch.mean(te_points,dim=0)
        dist = tr_ave - te_ave
        std = compute_std(gt_std,n_tr,n_te)
        loss = compute_gaussian_ot(dist,gt_std)
        return loss

    def compute_loss_oos_combo(tr_points,samples,ot_std,subsets_idx):

        # -- shape --
        T,D,BE = samples.shape
        device = tr_points.device
        
        # -- compute "model" --
        n_tr = len(tr_points)
        tr_ave = torch.mean(tr_points,dim=0)

        # -- create subsets to improve signal --
        T = samples.shape[0]
        ave = torch.mean(samples,dim=0)

        # -- compute ave diff between model and subsets --
        scores_t = torch.zeros(T,BE,device=device)
        counts_t = torch.zeros(T,1,device=device)
        for subset_idx in subsets_idx:
            nsub = T-len(subset_idx)+1
            counts_t[subset_idx] += 1
            subset = samples[subset_idx]
            vprint("subset.shape",subset.shape)
            subset_ave = torch.mean(subset,dim=0)
            vprint("subset_ave.shape",subset_ave.shape)
            n_sub = len(subset_idx)
            # loss = compute_pair_ave(tr_ave, subset_ave)
            # loss += compute_pair_ave(tr_ave, ave)
            # loss += compute_pair_ave(subset_ave, ave)
            # loss /= 3
            loss = compute_pair_ave(subset_ave, ave)
            loss /= (3 * nsub)
            # loss = compute_pair_mom(tr_ave, subset_ave, n_tr, n_sub)
            # dist = tr_ave - subset_ave
            # print_tensor_stats("tr_ave",tr_ave)
            # print_tensor_stats("subset_ave",subset_ave)
            # print_tensor_stats("dist",dist)
            # loss = compute_gaussian_ot(dist,ot_std)
            vprint("loss.shape",loss.shape)
            scores_t[subset_idx] += loss
        # n_evals = len(subsets_idx)
        # scores_t /= n_evals
        scores_t /= counts_t
        vprint(scores_t.shape)
        scores = torch.mean(scores_t,dim=0)
        scores_t = scores_t.T # (T,E) -> (E,T)
        vprint("scores.shape",scores.shape)
        return scores,scores_t

    def compute_train_est(tr_points):
        est = repeat(torch.mean(tr_points,dim=0),'be d -> n be d',n=te_N)

    # -=-=-=-=-=-=-=-=-=-=-=-
    #
    # -->      ransac     <--
    #
    # -=-=-=-=-=-=-=-=-=-=-=-

    # -- create subset grids --
    minSN,maxSN = 10,14
    indices = np.arange(T)
    max_subset_size = 600
    # subsets_idx = create_subset_grids_fixed(maxSN,indices,max_subset_size)
    subsets_idx = create_subset_grids(minSN,maxSN,indices,max_subset_size)

    # -- basic ransac hyperparams --
    desired_prob = .90
    # error_thresh = 1e-5
    size = 15 # 12 got (a) win and (b) match

    # -- set noise --
    gt_std = cfg.noise_params['g']['stddev']/255.
    ot_std = compute_std(gt_std,size,maxSN)
    ps_scale = (C*H*W*R)**(1/4.)
    # error_thresh = np.sqrt(gt_std**2/size + gt_std**2/maxSN) * 1.4 / ps_scale
    error_thresh = 1.
    # print(error_thresh)
    # error_thresh = np.sqrt(gt_std**2/size + gt_std**2/maxSN) * 1.96 / ps_scale
    # error_thresh = np.sqrt(gt_std**2/size + gt_std**2/maxSN) * 2.35 / ps_scale
    # print(gt_std)
    # print(ot_std)

    # -- init "bests" --
    scores_t = torch.zeros(B*E,T,device=device)
    scores = torch.zeros(B*E,device=device)
    best_scores = torch.ones(B*E,device=device) * float('inf')
    best_scores_t = torch.ones(B*E,T,device=device) * float('inf')

    # -- loop params --
    best_model = torch.zeros(B*E,T).int()
    best_model[:,t_ref] = 1
    # iters,num_iters,max_iters = 0,1000,1000
    # iters,num_iters,max_iters = 0,500,1000
    iters,num_iters,max_iters = 0,1,1000
    while iters < num_iters and iters < max_iters:

        # -- randomly select points with t_ref in train --
        order = torch.randperm(T)
        i_ref = torch.argmin(torch.abs(order - t_ref))
        order[i_ref] = order[0]
        order[0] = t_ref

        # -- split data --
        tr_index,te_index = order[:size],order[size:]
        tr_points,tr_N = samples[tr_index],len(tr_index)
        te_points,te_N = samples[te_index],len(te_index)
        # print(tr_index,te_index,best_model[0])
        
        # -- compute model --
        # loss_oos = compute_loss_oos(tr_points,samples,gt_std)
        # scores,scores_t = compute_loss_oos_combo(tr_points,te_points,ot_std,subsets_idx)
        losses,losses_t = compute_loss_oos_combo(tr_points,samples,ot_std,subsets_idx)
        # print(losses_t[:3,:])
        # print(losses_t[98,:])

        # -- adaptive threshold --
        # losses_std = torch.std(losses_t,dim=1)
        # losses_ave = torch.mean(losses_t,dim=1)
        # error_thresh = losses_ave + 1.96 * losses_std
        # print(error_thresh.shape,losses_t.shape)
        # print(losses_t)
        # print(losses_t[98])
        # exit()

        # -- check if best model --
        outliers = losses_t > error_thresh
        n_outliers = torch.sum(outliers,dim=1)
        # scores_t = outliers.float()
        # scores = torch.mean(scores_t,dim=1).float()
        args = torch.where(outliers)
        scores_t = losses_t.clone()
        scores_t[args[0],args[1]] += 1./T

        # inliers = losses_t < error_thresh
        # print(inliers.shape)
        # n_inliers = torch.sum(inliers,dim=1)
        # args = torch.where(inliers)
        # scores_t[args[0],args[1]] = losses_t[args[0],args[1]]
        # print(args[0].shape,args[1].shape,losses_t[args[0],args[1]].shape)
        # scores = torch.mean(scores_t,dim=1) * (10*torch.std(losses_t,dim=1))
        scores = torch.mean(scores_t,dim=1)
        # scores = torch.max(losses_t,dim=1).values - torch.min(losses_t,dim=1).values
        # print(scores[98],torch.mean(losses_t[98,:]))
        # print(scores[:3])
        # exit()
        
        # if inlier_count > max_inlier_count:
        #     max_inlier_count = inlier_count
        #     best_model = tr_index

        """
        We should not update our "best score"
        based on the average of _all_ samples
        
        We should update our score based on
        some component of inlier v.s. outliers
        
        An example includes the average score 
        of just the inliers...
        """
        # -- save best model for each batch --
        args = torch.where(scores < best_scores)[0]
        # if torch.any((args - 98) == 0):
        #     print("HI\n\n\n\n\n")
        #     print(scores[98])
        #     print("HI\n\n\n\n\n")
        nargs = len(args)
        args_rep = repeat(args,'nargs -> nargs n',n=len(tr_index))
        tr_index_rep = repeat(tr_index,'n -> nargs n',nargs=nargs)
        te_index_rep = repeat(te_index,'n -> nargs n',nargs=nargs)
        if len(args) > 0:
            best_scores[args] = scores[args]
            best_scores_t[args] = scores_t[args]
            best_model[args] = 0
            best_model[args_rep,tr_index_rep] = 1

        # -- update outlier prob --
        # prob_outlier = 1 - inlier_count/T
        # num_iters = math.log(1 - desired_prob)/math.log(1 - (1 - prob_outlier)**tr_size)
        iters = iters + 1

    # -- no cuda --
    scores = rearrange(best_scores.cpu(),'(b e) -> b e',b=B)
    scores_t = rearrange(best_scores_t.cpu(),'(b e) t -> b e t',b=B)
    best_model = rearrange(best_model.cpu(),'(b e) t -> b e t',b=B)

    bgrid = torch.arange(B)
    args = torch.argmin(scores,dim=1)

    # print(error_thresh)
    # print(args.shape)
    # print("argmin",args)
    # tgt_index = 98
    # print(best_model[:,tgt_index])
    # print(scores[:,tgt_index])
    # print(scores[bgrid,args])
    # print(scores[:,tgt_index] <= scores[bgrid,args])
    # exit()

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    return scores,scores_t

def ave_consistency(cfg,expanded):
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    # print("-> Print Time <-","\n\n\n")
    R,B,E,T,C,H,W = expanded.shape
    # expanded = expanded[[0]]
    expanded = rearrange(expanded,'r b e t c h w -> (b e) t (r c h w)')
    ave = torch.mean(expanded,dim=1)
    scores_t = []
    diffs = []
    t_ref = T//2
    for t in range(T):
        expanded_no_t = torch.cat([expanded[:,:t],expanded[:,t+1:]],dim=1)
        leave_out_t = torch.mean(expanded_no_t,dim=1)
        diffs.append(leave_out_t)
        # diffs_t = F.mse_loss(ave,leave_out_t,reduction='none')
        # diffs_t = T*(ave - leave_out_t)
        # diffs_t = expanded[:,t_ref,:] - expanded[:,t,:]
        # diffs.append(diffs_t)
        # ave_t = torch.mean(diffs_t,dim=1)
        # ac_vec_t,ac_coeff_t = torch_xcorr(diffs_t)
        # corr_term = -torch.log(torch.abs(ac_vec_t[:,0]))/10.
        # scores_t.append( torch.abs(ave_t) )#+  corr_term)
    # scores_t = torch.stack(scores_t,dim=1)

    def compare_diffs(x,y):
        mom_1 = (torch.mean(x,dim=1) - torch.mean(y,dim=1))**2
        mom_2 = (torch.std(x,dim=1)**2 - torch.std(y,dim=1)**2)**2
        return (mom_1 + mom_2).cpu()

    def compare_diffs_ot(x,y):
        x = rearrange(x,'be (r c hw) -> be (r hw) c',r=R,c=3)
        y = rearrange(y,'be (r c hw) -> be (r hw) c',r=R,c=3)
        dists = []
        BE = x.shape[0]
        for be in range(BE):
            M_be = dmat(x[be],y[be])
            # M_be = pairwise_distances(x[be],y[be])
            M_be = M_be.to(x.device)
            dist = sink_stabilized(M_be,reg=1.0,device=x.device).item()
            dists.append(dist)
        return torch.FloatTensor(dists)

    def compare_to_known(x,gt_std):
        mom_1 = (torch.mean(x,dim=1) - 0)**2
        mom_2 = (torch.std(x,dim=1) - gt_std)**2
        # ac_vec_t,ac_coeff_t = torch_xcorr(diffs_t)
        # rev_x = torch.flip(x,(1,))
        # xcorr = torch.einsum('bi,bj->b',x,rev_x)
        # ac_coeff_t = torch.mean(torch.abs(ac_vec_t[:,1:2]),dim=1)
        # print(ac_vec_t[:,0])
        # return (mom_1 + mom_2).cpu()# + ac_coeff_t).cpu()
        return (mom_1 + mom_2 + xcorr).cpu()# + ac_coeff_t).cpu()
        # return (mom_1 + mom_2 + ac_coeff_t).cpu()

    # print(ave.shape)
    # print(torch.mean(ave[0]),torch.std(ave[0]))
    # print(torch.mean(ave[1]),torch.std(ave[1]))

    cmps = []
    cmps = {str(t):0 for t in range(T)}
    # scores_t = torch.zeros((R*B*E,T))
    scores_t = torch.zeros((B*E,T))
    for t_i in range(T):
        for t_j in range(T):
            if t_i >= t_j: continue
            #comp_t_ij = compare_diffs(diffs[t_i],diffs[t_j])
            comp_t_ij = torch.sum(torch.abs(diffs[t_i]-diffs[t_j]),dim=1).cpu()
            scores_t[:,t_i] += comp_t_ij
            scores_t[:,t_j] += comp_t_ij
            # cmps[str(t_i)] += 1
            # cmps[str(t_j)] += 1
            # cmps.append(comp_t_ij)
    scores = torch.mean(scores_t,dim=1)
    scores_t = repeat(rearrange(scores_t,'(b e) t -> b e t',b=B,e=E),
                      'b e t -> r b e t',r=R)
    scores = repeat(rearrange(scores,'(b e) -> b e',b=B,e=E),
                    'b e -> r b e',r=R)
    # -- to cpu --
    scores = scores.cpu()
    scores_t = scores_t.cpu()

    return scores,scores_t

def sim_trm(cfg,expanded):
    vprint("-> Print Time <-","\n\n\n")
    R,B,E,T,C,H,W = expanded.shape
    # expanded = expanded[[0]]
    expanded = rearrange(expanded,'r b e t c h w -> (b e) t (r c h w)')
    ave = torch.mean(expanded,dim=1)
    scores_t = []
    diffs = []
    t_ref = T//2
    for t in range(T):
        expanded_no_t = torch.cat([expanded[:,:t],expanded[:,t+1:]],dim=1)
        leave_out_t = torch.mean(expanded_no_t,dim=1)
        # diffs.append(leave_out_t)
        # diffs_t = F.mse_loss(ave,leave_out_t,reduction='none')
        # diffs_t = T*(ave - leave_out_t)
        # diffs_t = expanded[:,t_ref,:] - expanded[:,t,:]
        # diffs.append(diffs_t)
        # scores_t.append( torch.abs(ave_t) )#+  corr_term)
        scores_t.append( torch.mean(leave_out_t - 0.5,dim=1).cpu() )#+  corr_term)
    scores_t = torch.stack(scores_t,dim=1)

    ave = rearrange(ave,'(b e) d -> b e d',b=B)
    scores = torch.zeros((B,E))
    for e_i in range(E):
        for e_j in range(E):
            if e_i >= e_j: continue
            l2_diff = torch.mean((ave[:,e_i] - ave[:,e_j])**2,dim=1).cpu()
            scores[:,e_i] += l2_diff
            scores[:,e_j] += l2_diff

    # scores = torch.mean(scores_t,dim=1)

    # -- be -> b e --
    scores_t = rearrange(scores_t,'(b e) t -> b e t',b=B,e=E)
    # scores = rearrange(scores,'(b e) -> b e',b=B,e=E)

    # -- repeat --
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)
    scores = repeat(scores,'b e -> r b e',r=R)

    # -- to cpu --
    scores = scores.cpu()
    scores_t = scores_t.cpu()
    print(torch.argmin(scores[0],1))

    return scores,scores_t

def jackknife(cfg,expanded):
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    vprint("-> Print Time <-","\n\n\n")
    R,B,E,T,C,H,W = expanded.shape
    # expanded = expanded[[0]]
    expanded = rearrange(expanded,'r b e t c h w -> (b e) t (r c h w)')
    ave = torch.mean(expanded,dim=1)
    scores_t = []
    diffs = []
    t_ref = T//2
    for t in range(T):
        expanded_no_t = torch.cat([expanded[:,:t],expanded[:,t+1:]],dim=1)
        leave_out_t = torch.mean(expanded_no_t,dim=1)
        # diffs.append(leave_out_t)
        # diffs_t = F.mse_loss(ave,leave_out_t,reduction='none')
        diffs_t = T*(ave - leave_out_t)
        # diffs_t = expanded[:,t_ref,:] - expanded[:,t,:]
        diffs.append(diffs_t)
        # ave_t = torch.mean(diffs_t,dim=1)
        # ac_vec_t,ac_coeff_t = torch_xcorr(diffs_t)
        # corr_term = -torch.log(torch.abs(ac_vec_t[:,0]))/10.
        # scores_t.append( torch.abs(ave_t) )#+  corr_term)
        # scores_t.append( torch.mean(leave_out_t - 0.5,dim=1).cpu() )#+  corr_term)
    # scores_t = torch.stack(scores_t,dim=1)

    def compare_diffs(x,y):
        mom_1 = (torch.mean(x,dim=1) - torch.mean(y,dim=1))**2
        mom_2 = (torch.std(x,dim=1)**2 - torch.std(y,dim=1)**2)**2
        return (mom_1 + mom_2).cpu()

    def compare_diffs_ot(x,y):
        x = rearrange(x,'be (r c hw) -> be (r hw) c',r=R,c=3)
        y = rearrange(y,'be (r c hw) -> be (r hw) c',r=R,c=3)
        dists = []
        BE = x.shape[0]
        for be in range(BE):
            M_be = dmat(x[be],y[be])
            # M_be = pairwise_distances(x[be],y[be])
            M_be = M_be.to(x.device)
            dist = sink_stabilized(M_be,reg=1.0,device=x.device).item()
            dists.append(dist)
        return torch.FloatTensor(dists)

    def compare_to_known(x,gt_std):
        mom_1 = (torch.mean(x,dim=1) - 0)**2
        mom_2 = (torch.std(x,dim=1) - gt_std)**2
        # ac_vec_t,ac_coeff_t = torch_xcorr(diffs_t)
        # rev_x = torch.flip(x,(1,))
        # xcorr = torch.einsum('bi,bj->b',x,rev_x)
        # ac_coeff_t = torch.mean(torch.abs(ac_vec_t[:,1:2]),dim=1)
        # print(ac_vec_t[:,0])
        return (mom_1 + mom_2).cpu()# + ac_coeff_t).cpu()
        # return (mom_1 + mom_2 + xcorr).cpu()# + ac_coeff_t).cpu()
        # return (mom_1 + mom_2 + ac_coeff_t).cpu()

    # print(ave.shape)
    # print(torch.mean(ave[0]),torch.std(ave[0]))
    # print(torch.mean(ave[1]),torch.std(ave[1]))

    # cmps = []
    # cmps = {str(t):0 for t in range(T)}
    # # scores_t = torch.zeros((R*B*E,T))
    # scores_t = torch.zeros((B*E,T))
    # for t_i in range(T):
    #     for t_j in range(T):
    #         if t_i >= t_j: continue
    #         #comp_t_ij = compare_diffs(diffs[t_i],diffs[t_j])
    #         comp_t_ij = torch.sum(torch.abs(diffs[t_i]-diffs[t_j]),dim=1).cpu()
    #         scores_t[:,t_i] += comp_t_ij
    #         scores_t[:,t_j] += comp_t_ij
    #         # cmps[str(t_i)] += 1
    #         # cmps[str(t_j)] += 1
    #         # cmps.append(comp_t_ij)

    t_ref = T//2
    noise = cfg.noise_params['g']['stddev']/255.
    gt_std = np.sqrt((noise**2 + noise**2/(T-1)))
    # scores_t = torch.zeros((R*B*E,T))
    # vprint(ave.shape)
    # for i in range(10):
    #     vprint('ave',torch.mean(ave,dim=1)[i],torch.std(ave,dim=1)[i],gt_std)
    #     for t in range(T):
    #         vprint(t,torch.mean(diffs[t],dim=1)[i],torch.std(diffs[t],dim=1)[i],gt_std)

    scores_t = torch.zeros((B*E,T))
    for t in range(T):
        #print(t,torch.mean(diffs[t],dim=1)[0],torch.std(diffs[t],dim=1)[0],gt_std)
        # comp_t = compare_diffs(diffs[t],diffs[t_ref])
        # comp_t = compare_diffs_ot(diffs[t],diffs[t_ref])
        comp_t = compare_to_known(diffs[t],gt_std)
        # comp_t = torch.mean((diffs[t] - diffs[t_ref])**2,dim=1).cpu()
        scores_t[:,t] += comp_t

    # print(cmps)
    # print(scores_t)
    scores = torch.mean(scores_t,dim=1)

    scores_t = repeat(rearrange(scores_t,'(b e) t -> b e t',b=B,e=E),
                      'b e t -> r b e t',r=R)
    scores = repeat(rearrange(scores,'(b e) -> b e',b=B,e=E),
                    'b e -> r b e',r=R)

    # -- to cpu --
    scores = scores.cpu()
    scores_t = scores_t.cpu()

    # scores_t = rearrange(scores_t,'(r b e) t -> r b e t',r=R,b=B,e=E)
    # scores = rearrange(scores,'(r b e) -> r b e',r=R,b=B,e=E)
    # print(scores.shape)
    # print(torch.argmin(scores[0],1))

    return scores,scores_t
    
def mse_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    # -- R goes to CHW --
    expanded = rearrange(expanded,'r b e t c h w -> t b e (r c h w)')
    
    ref = expanded[T//2]
    neighbors = expanded
    ave = torch.mean(expanded,dim=0)

    delta = torch.mean( (ave-ref)**2, dim=-1) * (T-1)/T
    
    # var_term = (neighbors - ref)**2
    # var_term = torch.mean(var_term,dim=(0,-1))

    # bias_term = (torch.mean(ave - ref,dim=-1))**2
    # bias_term = torch.mean(bias_term,dim=0)

    # delta = var_term + bias_term

    # -- repeat to include R --
    delta_t = torch.zeros(B,E,T)
    delta_t = repeat(delta_t,'b e t -> r b e t',r=R)
    delta = repeat(delta,'b e -> r b e',r=R)

    return delta,delta_t

def ave_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    # -- R goes to CHW --
    expanded = rearrange(expanded,'r b e t c h w -> t b e (r c h w)')
    
    # ref = repeat(expanded[T//2],'b e d -> tile b e d',tile=T-1)
    # neighbors = torch.cat([expanded[:T//2],expanded[T//2+1:]],dim=0)

    ref = expanded[T//2]
    neighbors = expanded

    # delta = F.mse_loss(ref,neighbors,reduction='none')
    delta = (ref - neighbors)**2 
    delta_t = torch.mean(delta,dim=3)
    delta = torch.mean(delta_t,dim=0)

    # -- append dim for T --
    if delta_t.shape[0] == T-1:
        delta_t = rearrange(delta_t,'t b e -> b e t')
        zeros = torch.zeros_like(delta_t[:,:,[0]])
        delta_t = torch.cat([delta_t[:,:,:T//2],zeros,delta_t[:,:,T//2:]],dim=2)
        # print("delta_t.shape",delta_t.shape)
    else:
        delta_t = rearrange(delta_t,'t b e -> b e t')

    # -- repeat to include R --
    delta_t = repeat(delta_t,'b e t -> r b e t',r=R)
    delta = repeat(delta,'b e -> r b e',r=R)

    return delta,delta_t


def ave_score_original(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    """
    R = different patches from same image
    B = different images
    E = differnet block regions around centered patch
    T = burst of frames along a batch dimension
    """
    ref = repeat(expanded[:,:,:,T//2],'r b e c h w -> r b e tile c h w',tile=T-1)
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta = F.mse_loss(ref,neighbors,reduction='none')
    delta = delta.view(R,B,E,T-1,-1)
    delta_t = torch.mean(delta,dim=4)
    delta = torch.mean(delta_t,dim=3)

    # -- append dim for T --
    Tm1 = T-1
    zeros = torch.zeros_like(delta_t[:,:,:,[0]])
    delta_t = torch.cat([delta_t[:,:,:,:Tm1//2],zeros,delta_t[:,:,:,Tm1//2:]],dim=3)

    return delta,delta_t

def refcmp_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    ref = expanded[:,:,:,T//2]
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta_t = torch.zeros(R,B,E,T,device=expanded.device)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for t in range(T-1):
        delta_pair = F.mse_loss(neighbors[:,:,:,t],ref,reduction='none')
        delta_t += delta_pair
        delta += torch.mean(delta_t.view(R,B,E,-1),dim=3)
    delta_t /= (T-1)
    delta /= (T-1)
    return delta,delta_t

def pairwise_delta_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape
    # ref = repeat(expanded[:,:,:,[T//2]],'r b e c h w -> r b e tile c h w',tile=T-1)
    # neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta_t = torch.zeros(R,B,E,T,device=expanded.device)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for t1 in range(T):
        for t2 in range(T):
            delta_pair = F.mse_loss(expanded[:,:,:,t1],expanded[:,:,:,t2],reduction='none')
            delta_t[:,:,:,[t1,t2]] += delta_pair
            delta += torch.mean(delta_t.view(R,B,E,-1),dim=3)
    delta /= T*T
    delta_t /= T*T
    return delta,delta_t

#
# Grid Functions
#

# -- run over the grids for below --
def delta_over_grids(cfg,expanded,grids):
    R,B,E,T,C,H,W = expanded.shape
    unrolled = rearrange(expanded,'r b e t c h w -> r b e t (c h w)')
    delta_t = torch.zeros(R,B,E,T,device=expanded.device)
    delta = torch.zeros(R,B,E,device=expanded.device)
    for set0,set1 in grids:
        set0,set1 = np.atleast_1d(set0),np.atleast_1d(set1)

        # -- compute ave --
        ave0 = torch.mean(expanded[:,:,:,set0],dim=3)
        ave1 = torch.mean(expanded[:,:,:,set1],dim=3)

        # -- rearrange --
        ave0 = rearrange(ave0,'r b e c h w -> r b e (c h w)')
        ave1 = rearrange(ave1,'r b e c h w -> r b e (c h w)')

        # -- rep across time --
        ave0_repT = repeat(ave0,'r b e f -> r b e t f',t=T)
        ave1_repT = repeat(ave1,'r b e f -> r b e t f',t=T)

        # -- compute deltas --
        delta_pair = F.mse_loss(ave0,ave1,reduction='none').view(R,B,E,-1)
        delta_0 = F.mse_loss(ave0_repT,unrolled,reduction='none').view(R,B,E,T,-1)
        delta_1 = F.mse_loss(ave1_repT,unrolled,reduction='none').view(R,B,E,T,-1)
        delta_t += torch.mean( (delta_0 + delta_1)/2., dim = 4)
        delta += torch.mean(delta_pair,dim=3)
    delta /= len(grids)
    delta_t /= len(grids)
    return delta,delta_t

def powerset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- powerset --
    indices = np.arange(T)
    powerset = chain.from_iterable(combinations(list(indices) , r+1 ) for r in range(T))
    powerset = np.array([np.array(elem) for elem in list(powerset)])
    grids = np.array(np.meshgrid(*[powerset,powerset]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def extrema_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- extrema subsets --
    indices = np.arange(T)
    subset_lg = create_combination(indices,T-2,T)
    subset_sm = create_combination(indices,0,2)
    subset_ex = np.r_[subset_sm,subset_lg]
    grids = np.array(np.meshgrid(*[subset_ex,subset_ex]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def smsubset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- compare large subsets --
    indices = np.arange(T)
    subset_sm = create_combination(indices,1,2)
    grids = np.array(np.meshgrid(*[subset_sm,subset_sm]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def lgsubset_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- compare large subsets --
    indices = np.arange(T)
    subset_lg = create_combination(indices,T-2,T)
    grids = np.array(np.meshgrid(*[subset_lg,subset_lg]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t


def lgsubset_v_indices_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    l_indices = [[i] for i in indices]
    l_indices.append(list(np.arange(T)))
    subset_lg = create_combination(indices,T-2,T)
    grids = np.array(np.meshgrid(*[subset_lg,l_indices]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def lgsubset_v_ref_score(cfg,expanded,ref_t=None):
    R,B,E,T,C,H,W = expanded.shape
    if ref_t is None: ref_t = T//2

    # -- indices and large subset --
    indices = np.arange(T)
    subset_lg = create_combination(indices,T-2,T)
    grids = np.array(np.meshgrid(*[subset_lg,[[ref_t,]]]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def powerset_v_indices_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    l_indices = [[i] for i in indices]
    l_indices.append(list(np.arange(T)))
    powerset = create_combination(indices,0,T)
    grids = np.array(np.meshgrid(*[powerset,l_indices]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

def powerset_v_ref_score(cfg,expanded):
    R,B,E,T,C,H,W = expanded.shape

    # -- indices and large subset --
    indices = np.arange(T)
    powerset = create_combination(indices,0,T)
    grids = np.array(np.meshgrid(*[powerset,[T//2]]))
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- compute loss --
    delta,delta_t = delta_over_grids(cfg,expanded,grids)
    return delta,delta_t

#
# Optimal Transport Based Losses
# 

def gaussian_ot_score(cfg,expanded,return_frames=False):
    R,B,E,T,C,H,W = expanded.shape
    vectorize = rearrange(expanded,'r b e t c h w -> (r b e t) (c h w)')
    means = torch.mean(vectorize,dim=1)
    stds = torch.std(vectorize,dim=1)

    # -- gaussian zero mean, var = noise_level --
    gt_std = cfg.noise_params['g']['stddev']/255.
    loss = means**2
    loss += (stds**2 - 2*gt_std**2)**2
    losses_t = rearrange(loss,'(r b e t) -> r b e t',r=R,b=B,e=E,t=T)
    losses = torch.mean(losses_t,dim=3)
    return losses,losses_t

def emd_score(cfg,expanded):
    pass

