
# -- python imports --
import time
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
import settings
from pyutils import tile_patches,save_image,torch_to_numpy
from pyutils.vst import anscombe
from patch_search import get_score_function
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

# -- [align] package imports --
from align import compute_epe,compute_aligned_psnr
import align.nnf as nnf
from align.combo.eval_scores import EvalBlockScores
from align.combo.optim import AlignOptimizer
from align.xforms import align_from_pix,flow_to_pix,create_isize,pix_to_flow,align_from_flow,flow_to_blocks

# -- cuda profiler --
import nvtx

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def remove_center_frame(frames):
    nframes = frames.shape[0]
    nc_frames =torch.cat([frames[:nframes//2],frames[nframes//2+1:]],dim=0)
    return nc_frames

def check_parameters(nblocks,patchsize):
    even_blocks = nblocks % 2 == 0
    even_ps = patchsize % 2 == 0
    if even_blocks or even_ps:
        print("Even number of blocks or patchsizes. We recommend odd so a center exists.")

@nvtx.annotate("test_nnf", color="purple")
def execute_experiment(cfg):

    # -- init --
    print("RUNNING EXP.")
    print(cfg)

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load dataset --
    data,loaders = load_image_dataset(cfg)
    image_iter = iter(loaders.tr)    
    nskips = 2+4+2+4+1
    for skip in range(nskips): next(image_iter)
    
    # -- get score function --
    score_fxn_ave = get_score_function("ave")
    score_fxn_bs = get_score_function(cfg.score_fxn_name)

    # -- some constants --
    NUM_BATCHES = 10
    nframes,nblocks = cfg.nframes,cfg.nblocks 
    patchsize = cfg.patchsize
    ppf = cfg.dynamic_info.ppf
    check_parameters(nblocks,patchsize)

    # -- create evaluator for ave; simple --
    iterations,K = 1,1
    subsizes = []
    block_batchsize = 256
    eval_ave_simp = EvalBlockScores(score_fxn_ave,patchsize,block_batchsize,None)

    # -- create evaluator for ave --
    iterations,K = 1,1
    subsizes = []
    eval_ave = EvalBlockScores(score_fxn_ave,patchsize,block_batchsize,None)

    # -- create evaluator for bootstrapping --
    block_batchsize = 64
    eval_prop = EvalBlockScores(score_fxn_bs,patchsize,block_batchsize,None)

    # -- iterate over images --
    for image_bindex in range(NUM_BATCHES):

        print("-="*30+"-")
        print(f"Running image batch index: {image_bindex}")
        print("-="*30+"-")

        # -- sample & unpack batch --
        sample = next(image_iter)
        sample_to_cuda(sample)

        dyn_noisy = sample['noisy'] # dynamics and noise
        dyn_clean = sample['burst'] # dynamics and no noise
        static_noisy = sample['snoisy'] # no dynamics and noise
        static_clean = sample['sburst'] # no dynamics and no noise
        flow_gt = sample['flow']
        if cfg.noise_params.ntype == "pn":
            dyn_noisy = anscombe.forward(dyn_noisy)

        # -- shape info --
        T,B,C,H,W = dyn_noisy.shape
        isize = edict({'h':H,'w':W})
        ref_t = nframes//2
        npix = H*W

        # -- groundtruth flow --
        # print("flow_gt",flow_gt)
        flow_gt_rs = rearrange(flow_gt,'i tm1 two -> i 1 tm1 two')
        blocks_gt = flow_to_blocks(flow_gt_rs,nblocks)
        # print("\n\n")
        # print("flow_gt[0,0] ",flow_gt)
        # print("blocks_gt[0,0] ",blocks_gt[0,0])
        flow_gt = repeat(flow_gt,'i tm1 two -> i p tm1 two',p=npix)
        aligned_of = align_from_flow(dyn_clean,flow_gt,nblocks,isize=isize)
        pix_gt = flow_to_pix(flow_gt.clone(),isize=isize)
        
        # -- compute nearest neighbor fields --
        start_time = time.perf_counter()
        shape_str = 't b h w two -> b (h w) t two'
        nnf_vals,nnf_pix = nnf.compute_burst_nnf(dyn_clean,ref_t,patchsize)
        nnf_pix_best = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        nnf_pix_best = torch.LongTensor(nnf_pix_best)
        pix_nnf = nnf_pix_best.clone()
        flow_nnf = pix_to_flow(nnf_pix_best)
        aligned_nnf = align_from_pix(dyn_clean,nnf_pix_best,nblocks)
        time_nnf = time.perf_counter() - start_time

        # -- compute proposed search of nnf --
        start_time = time.perf_counter()
        print(dyn_noisy.shape)
        # split_vals,split_pix = nnf.compute_burst_nnf(dyn_noisy,ref_t,patchsize)
        split_pix = np.copy(nnf_pix)
        split_pix_best = torch.LongTensor(rearrange(split_pix[...,0,:],shape_str))
        split_pix_best = torch.LongTensor(split_pix_best)
        pix_split = split_pix_best.clone()
        flow_split = pix_to_flow(split_pix_best)
        aligned_split = align_from_pix(dyn_clean,split_pix_best,nblocks)
        time_split = time.perf_counter() - start_time

        # -- compute simple ave --
        iterations,K = 0,1
        subsizes = []
        print("[simple] Ave loss function")
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        # flow_ave_simp = optim.run(dyn_noisy,patchsize,eval_ave_simp,
        #                      nblocks,iterations,subsizes,K)
        flow_ave_simp = flow_gt.clone().cpu()
        aligned_ave_simp = align_from_flow(dyn_clean,flow_ave_simp,nblocks,isize=isize)
        time_ave_simp = time.perf_counter() - start_time
        print(flow_ave_simp.shape)

        # -- compute complex ave --
        iterations,K = 0,1
        subsizes = []
        print("[complex] Ave loss function")
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        flow_ave = optim.run(dyn_noisy,patchsize,eval_ave,
                             nblocks,iterations,subsizes,K)
        # flow_ave = flow_gt.clone()
        pix_ave = flow_to_pix(flow_ave.clone(),isize=isize)
        aligned_ave = align_from_flow(dyn_clean,flow_ave,nblocks,isize=isize)
        time_ave = time.perf_counter() - start_time

        # -- compute proposed search of nnf --
        # iterations,K = 50,3
        # subsizes = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
        #iterations,K = 1,nblocks**2
        # K is a function of noise level.
        # iterations,K = 1,nblocks**2
        iterations,K = 1,2*nblocks#**2
        # subsizes = [3]#,3,3,3,3,3,3,3,3,3]
        # subsizes = [3,3,3,3,3,3,3,]
        subsizes = [3,3,3,3,3,3,3,3]
        # subsizes = [nframes]
        # subsizes = [nframes]
        print("[Bootstrap] loss function")
        start_time = time.perf_counter()
        optim = AlignOptimizer("v3")
        flow_est = optim.run(dyn_noisy,patchsize,eval_prop,
                             nblocks,iterations,subsizes,K)
        pix_est = flow_to_pix(flow_est.clone(),isize=isize)
        aligned_est = align_from_flow(dyn_clean,flow_est,patchsize,isize=isize)
        time_est = time.perf_counter() - start_time
        # flow_est = flow_gt.clone()
        # aligned_est = aligned_of.clone()
        # time_est = 0.

        # -- banner --
        print("\n"*3)
        print("-"*25 + " Results " + "-"*25)

        # -- examples of flow --
        print("-"*50)
        is_even = cfg.frame_size%2 == 0
        mid_pix = cfg.frame_size*cfg.frame_size//2 + (cfg.frame_size//2)*is_even
        mid_pix = 32*10+23
        # mid_pix = 32*23+10
        flow_gt_np = torch_to_numpy(flow_gt)
        flow_nnf_np = torch_to_numpy(flow_nnf)
        flow_split_np = torch_to_numpy(flow_split)
        flow_ave_simp_np = torch_to_numpy(flow_ave_simp)
        flow_ave_np = torch_to_numpy(flow_ave)
        flow_est_np = torch_to_numpy(flow_est)
        print(flow_gt_np[0,mid_pix])
        print(flow_nnf_np[0,mid_pix])
        print(flow_split_np[0,mid_pix])
        print(flow_ave_simp_np[0,mid_pix])
        print(flow_ave_np[0,mid_pix])
        print(flow_est_np[0,mid_pix])
        print("-"*50)
        pix_gt_np = torch_to_numpy(pix_gt)
        pix_nnf_np = torch_to_numpy(pix_nnf)
        pix_ave_np = torch_to_numpy(pix_ave)
        pix_est_np = torch_to_numpy(pix_est)
        print(pix_gt_np[0,mid_pix])
        print(pix_nnf_np[0,mid_pix])
        print(pix_ave_np[0,mid_pix])
        print(pix_est_np[0,mid_pix])

        # print(aligned_of[0,0,:,10,23].cpu() - static_clean[0,0,:,10,23].cpu())
        # print(aligned_ave[0,0,:,10,23].cpu() - static_clean[0,0,:,10,23].cpu())

        # print(aligned_of[0,0,:,23,10].cpu() - static_clean[0,0,:,23,10].cpu())
        # print(aligned_ave[0,0,:,23,10].cpu() - static_clean[0,0,:,23,10].cpu())

        print("-"*50)

        # -- compare compute time --
        print("-"*50)
        print("Compute Time [smaller is better]")
        print("-"*50)
        print("[NNF]: %2.3e" % time_nnf)
        print("[Split]: %2.3e" % time_split)
        print("[Ave [Simple]]: %2.3e" % time_ave_simp)
        print("[Ave]: %2.3e" % time_ave)
        print("[Proposed]: %2.3e" % time_est)

        # -- compare gt v.s. nnf computations --
        nnf_of = compute_epe(flow_nnf,flow_gt)
        split_of = compute_epe(flow_split,flow_gt)
        ave_simp_of = compute_epe(flow_ave_simp,flow_gt)
        ave_of = compute_epe(flow_ave,flow_gt)
        est_of = compute_epe(flow_est,flow_gt)

        split_nnf = compute_epe(flow_split,flow_nnf)
        ave_simp_nnf = compute_epe(flow_ave_simp,flow_nnf)
        ave_nnf = compute_epe(flow_ave,flow_nnf)
        est_nnf = compute_epe(flow_est,flow_nnf)

        # -- End-Point-Errors --
        print("-"*50)
        print("EPE Errors [smaller is better]")
        print("-"*50)

        print("NNF v.s. Optical Flow.")
        print(nnf_of)
        print("Split v.s. Optical Flow.")
        print(split_of)
        print("Ave [Simple] v.s. Optical Flow.")
        print(ave_simp_of)
        print("Ave v.s. Optical Flow.")
        print(ave_of)
        print("Proposed v.s. Optical Flow.")
        print(est_of)
        print("Split v.s. NNF")
        print(split_nnf)
        print("Ave [Simple] v.s. NNF")
        print(ave_simp_nnf)
        print("Ave v.s. NNF")
        print(ave_nnf)
        print("Proposed v.s. NNF")
        print(est_nnf)

        # -- compare accuracy of method nnf v.s. actual nnf --
        def compute_flow_acc(guess,gt):
            both = torch.all(guess.type(torch.long) == gt.type(torch.long),dim=-1)
            ncorrect  = torch.sum(both)
            acc = 100 * float(ncorrect) / both.numel()
            return acc 

        split_nnf_acc = compute_flow_acc(flow_split,flow_nnf)
        ave_simp_nnf_acc = compute_flow_acc(flow_ave_simp,flow_nnf)
        ave_nnf_acc = compute_flow_acc(flow_ave,flow_nnf)
        est_nnf_acc = compute_flow_acc(flow_est,flow_nnf)


        # -- PSNR to Reference Image --
        pad = 2*(nframes-1)*ppf+4
        isize = edict({'h':H-pad,'w':W-pad})
        # print("isize: ",isize)
        aligned_of = remove_center_frame(aligned_of)
        aligned_nnf = remove_center_frame(aligned_nnf)
        aligned_split = remove_center_frame(aligned_split)
        aligned_ave_simp = remove_center_frame(aligned_ave_simp)
        aligned_ave = remove_center_frame(aligned_ave)
        aligned_est = remove_center_frame(aligned_est)
        static_clean = remove_center_frame(static_clean)

        psnr_of = compute_aligned_psnr(aligned_of,static_clean,isize)
        psnr_nnf = compute_aligned_psnr(aligned_nnf,static_clean,isize)
        psnr_split = compute_aligned_psnr(aligned_split,static_clean,isize)
        psnr_ave_simp = compute_aligned_psnr(aligned_ave_simp,static_clean,isize)
        psnr_ave = compute_aligned_psnr(aligned_ave,static_clean,isize)
        psnr_est = compute_aligned_psnr(aligned_est,static_clean,isize)

        print("-"*50)
        print("PSNR Values [bigger is better]")
        print("-"*50)

        print("Optical Flow [groundtruth v1]")
        print(psnr_of)
        print("NNF [groundtruth v2]")
        print(psnr_nnf)
        print("Split [old method]")
        print(psnr_split)
        print("Ave [simple; old method]")
        print(psnr_ave_simp)
        print("Ave [old method]")
        print(psnr_ave)
        print("Proposed [new method]")
        print(psnr_est)


        # -- print nnf accuracy here --

        print("-"*50)
        print("NNF Accuracy [bigger is better]")
        print("-"*50)

        print("Split v.s. NNF")
        print(split_nnf_acc)
        print("Ave [Simple] v.s. NNF")
        print(ave_simp_nnf_acc)
        print("Ave v.s. NNF")
        print(ave_nnf_acc)
        print("Proposed v.s. NNF")
        print(est_nnf_acc)

        # -- location of PSNR errors --
        csize = 30
        # aligned_of = torch_to_numpy(tvF.center_crop(aligned_of,(csize,csize)))
        # aligned_ave = torch_to_numpy(tvF.center_crop(aligned_ave,(csize,csize)))
        # static_clean = torch_to_numpy(tvF.center_crop(static_clean,(csize,csize)))
        flow_gt = torch_to_numpy(flow_gt)
        flow_ave = torch_to_numpy(flow_ave)
        aligned_of = torch_to_numpy(aligned_of)
        aligned_ave = torch_to_numpy(aligned_ave)
        static_clean = torch_to_numpy(static_clean)

        # print("WHERE?")
        # print("OF")
        # print(aligned_of.shape)
        # for row in range(30):
        #     print(np.abs(aligned_of[0,0,0,row]- static_clean[0,0,0,row]))
        # print(np.where(~np.isclose(aligned_of,aligned_of)))
        # print(np.where(~np.isclose(flow_gt,flow_ave)))
        # print(np.where(~np.isclose(aligned_of,aligned_of)))
        # print(np.where(~np.isclose(aligned_of,static_clean)))
        # print("Ave")
        # indices = np.where(~np.isclose(aligned_ave,static_clean))
        # row,col = indices[-2:]
        # for elem in range(len(row)):
        #     print(np.c_[row,col][elem])
        # print(np.where(~np.isclose(aligned_ave,static_clean)))

        # -- Summary of End-Point-Errors --
        print("-"*50)
        print("Summary of EPE Errors [smaller is better]")
        print("-"*50)

        print("[NNF v.s. Optical Flow]: %2.3f" % nnf_of.mean().item())
        print("[Split v.s. Optical Flow]: %2.3f" % split_of.mean().item())
        print("[Ave [Simple] v.s. Optical Flow]: %2.3f" % ave_simp_of.mean().item())
        print("[Ave v.s. Optical Flow]: %2.3f" % ave_of.mean().item())
        print("[Proposed v.s. Optical Flow]: %2.3f" % est_of.mean().item())
        print("[Split v.s. NNF]: %2.3f" % split_nnf.mean().item())
        print("[Ave [Simple] v.s. NNF]: %2.3f" % ave_simp_nnf.mean().item())
        print("[Ave v.s. NNF]: %2.3f" % ave_nnf.mean().item())
        print("[Proposed v.s. NNF]: %2.3f" % est_nnf.mean().item())

        # -- Summary of PSNR to Reference Image --

        print("-"*50)
        print("Summary PSNR Values [bigger is better]")
        print("-"*50)

        print("[Optical Flow]: %2.3f" % psnr_of.mean().item())
        print("[NNF]: %2.3f" % psnr_nnf.mean().item())
        print("[Split]: %2.3f" % psnr_split.mean().item())
        print("[Ave [Simple]]: %2.3f" % psnr_ave_simp.mean().item())
        print("[Ave]: %2.3f" % psnr_ave.mean().item())
        print("[Proposed]: %2.3f" % psnr_est.mean().item())

        print("-"*50)
        print("PSNR Comparisons [smaller is better]")
        print("-"*50)
        delta_split = psnr_nnf - psnr_split
        delta_ave_simp = psnr_nnf - psnr_ave_simp
        delta_ave = psnr_nnf - psnr_ave
        delta_est = psnr_nnf - psnr_est
        print("ave([NNF] - [Split]): %2.3f" % delta_split.mean().item())
        print("ave([NNF] - [Ave [Simple]]): %2.3f" % delta_ave_simp.mean().item())
        print("ave([NNF] - [Ave]): %2.3f" % delta_ave.mean().item())
        print("ave([NNF] - [Proposed]): %2.3f" % delta_est.mean().item())

