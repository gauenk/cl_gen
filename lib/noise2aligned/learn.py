
# -- python imports --
import bm3d
import asyncio
import pandas as pd
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange, repeat, reduce

# -- just to draw an fing arrow --
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# -- faiss imports --
import faiss
import faiss.contrib.torch_utils

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

# -- project code --
import settings
from pyutils.timer import Timer
from pyutils import np_log,rescale_noisy_image,mse_to_psnr,save_image,tile_across_blocks,print_tensor_stats
from pyutils.vst import anscombe
from datasets.transforms import ScaleZeroMean,RandomChoice
from layers.ot_pytorch import sink_stabilized,sink,pairwise_distances
from layers.burst import BurstRecLoss,EntropyLoss
from datasets.transforms import get_noise_transform

# -- [local] project code --
from abps.abp_search import abp_search
from abps.check_abp_search import test_abp_global_search

from lpas import lpas_search,lpas_spoof

from n2nwl.dist_loss import ot_pairwise_bp,ot_gaussian_bp,ot_pairwise2gaussian_bp,kl_gaussian_bp,w_gaussian_bp,kl_gaussian_bp_patches
from n2nwl.misc import AlignmentFilterHooks
from n2nwl.plot import plot_histogram_residuals_batch,plot_histogram_gradients,plot_histogram_gradient_norms
from n2sim.sim_search import compute_similar_bursts,compute_similar_bursts_async,compute_similar_bursts_n2sim,create_k_grid,compare_sim_patches_methods,compare_sim_images_methods,compute_kindex_rands_async,kIndexPermLMDB,create_k_grid_v3
from .utils import crop_center_patch

# def print_tensor_stats(prefix,tensor):
#     stats_fmt = (tensor.min().item(),tensor.max().item(),tensor.mean().item())
#     stats_str = "%2.2e,%2.2e,%2.2e" % stats_fmt
#     print(prefix,stats_str)

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print("Hello World")

async def run_both(cfg,burst,K,patchsize,kindex):
    both = asyncio.gather(
        compute_kindex_rands_async(cfg,burst,K),
        compute_similar_bursts_async(cfg,burst,K,patchsize=patchsize,kindex=kindex)
    )
    await both
    return both

def sample_not_mid(N):
    not_mid = torch.LongTensor(np.r_[np.r_[:N//2],np.r_[N//2+1:N]])
    ones = torch.ones(N-1) / (N-1)
    idx = ones.multinomial(num_samples=1,replacement=False)[0]
    return not_mid[idx]
    
def shuffle_aligned_pixels_noncenter(aligned,R):
    T = aligned.shape[0]
    left_aligned,right_aligned = aligned[:T//2],aligned[T//2+1:]
    nc_aligned = torch.cat([left_aligned,right_aligned],dim=0)
    shuf = shuffle_aligned_pixels(nc_aligned,R)
    return shuf

def create_sim_from_aligned(burst,aligned,nsims):
    T = aligned.shape[0]
    shuffled = shuffle_aligned_pixels_noncenter(aligned,nsims)
    left,right = burst[:T//2],burst[T//2+1:]
    sim_aligned = []
    for r in range(nsims):
        sim = torch.cat([left,shuffled[[r]],right],dim=0)
        sim_aligned.append(sim)
    sim_aligned = torch.stack(sim_aligned,dim=0)
    sim_aligned = rearrange(sim_aligned,'r t b c h w -> t b r c h w')
    return sim_aligned

def shuffle_aligned_pixels(aligned,R):
    T,B,C,H,W = aligned.shape
    aligned = rearrange(aligned,'n b c h w -> n b c (h w)')
    shuffled = repeat(aligned[0].clone(),'b c hw -> r b c hw',r=R)
    hw_grid = torch.arange(H*W)
    for b in range(B):
        for r in range(R):
            indices = torch.randint(T,(H*W,)).long().to(aligned.device)
            for c in range(C):
                shuffled[r,b,c,:] = aligned[indices[r],b,c,hw_grid]
    shuffled = rearrange(shuffled,'r b c (h w) -> r b c h w',h=H)
    # aligned = rearrange(aligned,'n b c (h w) -> n b c h w',h=H)
    # images = [shuffled,aligned]
    # cropped = crop_center_patch(images,3,128)
    # shuffled,aligned = images[0],images[1]
    # print_tensor_stats("shuffled - aligned",shuffled - aligned)
    # print_tensor_stats("aligned0 - aligned1",aligned[0] - aligned[1])
    # exit()
    return shuffled

def train_loop(cfg,model,scheduler,train_loader,epoch,record_losses,writer):


    # -=-=-=-=-=-=-=-=-=-=-
    #
    #    Setup for epoch
    #
    # -=-=-=-=-=-=-=-=-=-=-

    model.align_info.model.train()
    model.denoiser_info.model.train()
    model.unet_info.model.train()
    model.denoiser_info.model = model.denoiser_info.model.to(cfg.device)
    model.align_info.model = model.align_info.model.to(cfg.device)
    model.unet_info.model = model.unet_info.model.to(cfg.device)


    N = cfg.N
    total_loss = 0
    running_loss = 0
    szm = ScaleZeroMean()
    blocksize = 128
    unfold = torch.nn.Unfold(blocksize,1,0,blocksize)
    use_record = False
    if record_losses is None: record_losses = pd.DataFrame({'burst':[],'ave':[],'ot':[],'psnr':[],'psnr_std':[]})
    noise_type = cfg.noise_params.ntype

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Init Record Keeping
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    align_mse_losses,align_mse_count = 0,0
    rec_mse_losses,rec_mse_count = 0,0
    rec_ot_losses,rec_ot_count = 0,0
    running_loss,total_loss = 0,0

    write_examples = False
    write_examples_iter = 200
    noise_level = cfg.noise_params['g']['stddev']

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #   Load Pre-Simulated Random Numbers
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
    if cfg.use_kindex_lmdb: kindex_ds = kIndexPermLMDB(cfg.batch_size,cfg.N)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Dataset Augmentation
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    transforms = [tvF.vflip,tvF.hflip,tvF.rotate]
    aug = RandomChoice(transforms)
    def apply_transformations(burst,gt_img):
        N,B = burst.shape[:2]
        gt_img_rs = rearrange(gt_img,'b c h w -> 1 b c h w')
        all_images = torch.cat([gt_img_rs,burst],dim=0)
        all_images = rearrange(all_images,'n b c h w -> (n b) c h w')
        tv_utils.save_image(all_images,'aug_original.png',nrow=N+1,normalize=True)
        aug_images = aug(all_images)
        tv_utils.save_image(aug_images,'aug_augmented.png',nrow=N+1,normalize=True)
        aug_images = rearrange(aug_images,'(n b) c h w -> n b c h w',b=B)
        aug_gt_img = aug_images[0]
        aug_burst = aug_images[1:]
        return aug_burst,aug_gt_img

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Half Precision
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # model.align_info.model.half()
    # model.denoiser_info.model.half()
    # model.unet_info.model.half()
    # models = [model.align_info.model,
    #           model.denoiser_info.model,
    #           model.unet_info.model]
    # for model_l in models:
    #     model_l.half()
    #     for layer in model_l.modules():
    #         if isinstance(layer, torch.nn.BatchNorm2d):
    #             layer.float()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Init Loss Functions
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    alignmentLossMSE = BurstRecLoss()
    denoiseLossMSE = BurstRecLoss(alpha=cfg.kpn_burst_alpha,gradient_L1=~cfg.supervised)
    # denoiseLossOT = BurstResidualLoss()
    entropyLoss = EntropyLoss()

    # -=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #    Add hooks for epoch
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-

    align_hook = AlignmentFilterHooks(cfg.N)
    align_hooks = []
    for kpn_module in model.align_info.model.children():
        for name,layer in kpn_module.named_children():
            if name == "filter_cls":
                align_hook_handle = layer.register_forward_hook(align_hook)
                align_hooks.append(align_hook_handle)
                

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #     Noise2Noise
    #
    # -=-=-=-=-=-=-=-=-=-=-

    noise_xform = get_noise_transform(cfg.noise_params,
                                      use_to_tensor=False)

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #    Final Configs
    #
    # -=-=-=-=-=-=-=-=-=-=-

    use_timer = False
    one = torch.FloatTensor([1.]).to(cfg.device)
    switch = True
    if use_timer:
        data_clock = Timer()
        clock = Timer()
    ds_size = len(train_loader)
    small_ds = ds_size < 500
    steps_per_epoch = ds_size if not small_ds else 500

    write_examples_iter = steps_per_epoch//3
    all_filters = []

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #     Start Epoch
    #
    # -=-=-=-=-=-=-=-=-=-=-

    init = torch.initial_seed()
    torch.manual_seed(cfg.seed+1+epoch+init)
    train_iter = iter(train_loader)
    for batch_idx in range(steps_per_epoch):

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #      Setting up for Iteration
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- setup iteration timer --
        if use_timer:
            data_clock.tic()
            clock.tic()

        # -- grab data batch --
        if small_ds and batch_idx >= ds_size:
            init = torch.initial_seed()
            torch.manual_seed(cfg.seed+1+epoch+init)
            train_iter = iter(train_loader) # reset if too big
        sample = next(train_iter)
        burst,raw_img,motion = sample['burst'],sample['clean'],sample['directions']
        raw_img_iid = sample['iid']
        raw_img_iid = raw_img_iid.cuda(non_blocking=True)
        burst = burst.cuda(non_blocking=True)

        # -- handle possibly cached simulated bursts --
        if 'sim_burst' in sample: sim_burst = rearrange(sample['sim_burst'],'b n k c h w -> n b k c h w')
        else: sim_burst = None
        non_sim_method = cfg.n2n or cfg.supervised
        if sim_burst is None and not (non_sim_method or cfg.abps):
            if sim_burst is None:
                if cfg.use_kindex_lmdb: kindex = kindex_ds[batch_idx].cuda(non_blocking=True)
                else: kindex = None
                query = burst[[N//2]]
                database = torch.cat([burst[:N//2],burst[N//2+1:]])
                sim_burst = compute_similar_bursts(cfg,query,database,
                                                   cfg.sim_K,noise_level/255.,
                                                   patchsize=cfg.sim_patchsize,
                                                   shuffle_k=cfg.sim_shuffleK,
                                                   kindex=kindex,only_middle=cfg.sim_only_middle,
                                                   search_method=cfg.sim_method,db_level="frame")
            
        if (sim_burst is None) and cfg.abps:
            # scores,aligned = abp_search(cfg,burst)
            # scores,aligned = lpas_search(cfg,burst,motion)
            mtype = "global"
            acc = cfg.optical_flow_acc
            scores,aligned = lpas_spoof(burst,motion,cfg.nblocks,mtype,acc)
            # scores,aligned = lpas_spoof(motion,accuracy=cfg.optical_flow_acc)
            # shuffled = shuffle_aligned_pixels_noncenter(aligned,cfg.nframes)
            nsims = cfg.nframes
            sim_aligned = create_sim_from_aligned(burst,aligned,nsims)
            burst_s = rearrange(burst,'t b c h w -> t b 1 c h w')
            sim_burst = torch.cat([burst_s,sim_aligned],dim=2)
            # print("sim_burst.shape",sim_burst.shape)

        # raw_img = raw_img.cuda(non_blocking=True)-0.5
        # # print(np.sqrt(cfg.noise_params['g']['stddev']))
        # print(motion)
        # tiled = tile_across_blocks(burst[[cfg.nframes//2]],cfg.nblocks)
        # rep_burst = repeat(burst,'t b c h w -> t b g c h w',g=tiled.shape[2])
        # for t in range(cfg.nframes):
        #     save_image(tiled[0] - rep_burst[t],f"tiled_sub_burst_{t}.png")
        # save_image(aligned,"aligned.png")
        # print(aligned.shape)
        # # save_image(aligned[0] - aligned[cfg.nframes//2],"aligned_0.png")
        # # save_image(aligned[2] - aligned[cfg.nframes//2],"aligned_2.png")
        # M = (1+cfg.dynamic.ppf)*cfg.nframes
        # fs = cfg.dynamic.frame_size - M
        # fs = cfg.frame_size
        # cropped = crop_center_patch([burst,aligned,raw_img],cfg.nframes,cfg.frame_size)
        # burst,aligned,raw_img = cropped[0],cropped[1],cropped[2]
        # print(aligned.shape)
        # for t in range(cfg.nframes+1):
        #     diff_t = aligned[t] - raw_img
        #     spacing = cfg.nframes+1
        #     diff_t = crop_center_patch([diff_t],spacing,cfg.frame_size)[0]
        #     print_tensor_stats(f"diff_aligned_{t}",diff_t)
        #     save_image(diff_t,f"diff_aligned_{t}.png")
        #     if t < cfg.nframes:
        #         dt = aligned[t+1]-aligned[t]
        #         dt = crop_center_patch([dt],spacing,cfg.frame_size)[0]
        #         save_image(dt,f"dt_aligned_{t+1}m{t}.png")
        #     save_image(aligned[t],f"aligned_{t}.png")
        #     diff_t = tvF.crop(aligned[t] - raw_img,cfg.nframes,cfg.nframes,fs,fs)
        #     print_tensor_stats(f"diff_aligned_{t}",diff_t)

        # save_image(burst,"burst.png")
        # save_image(burst[0] - burst[cfg.nframes//2],"burst_0.png")
        # save_image(burst[2] - burst[cfg.nframes//2],"burst_2.png")
        # exit()


        # print(sample['burst'].shape,sample['res'].shape)
        # b_clean = sample['burst'] - sample['res']
        # scores,ave,t_aligned = test_abp_global_search(cfg,b_clean,noisy_img=burst)

        # burstBN = rearrange(burst,'n b c h w -> (b n) c h w')
        # tv_utils.save_image(burstBN,"abps_burst.png",normalize=True)
        # alignedBN = rearrange(aligned,'n b c h w -> (b n) c h w')
        # tv_utils.save_image(alignedBN,"abps_aligned.png",normalize=True)
        # rep_burst = burst[[N//2]].repeat(N,1,1,1,1)
        # deltaBN = rearrange(aligned - rep_burst,'n b c h w -> (b n) c h w')
        # tv_utils.save_image(deltaBN,"abps_delta.png",normalize=True)
        # b_clean_rep = b_clean[[N//2]].repeat(N,1,1,1,1)
        # tdeltaBN = rearrange(t_aligned - b_clean_rep.cpu(),'n b c h w -> (b n) c h w')
        # tv_utils.save_image(tdeltaBN,"abps_tdelta.png",normalize=True)


        if non_sim_method: sim_burst = burst.unsqueeze(2).repeat(1,1,2,1,1,1)
        else: sim_burst = sim_burst.cuda(non_blocking=True)
        if use_timer: data_clock.toc()
    
        # -- to cuda --
        burst = burst.cuda(non_blocking=True)
        raw_zm_img = szm(raw_img.cuda(non_blocking=True))
        # anscombe.test(cfg,burst_og)
        # save_image(burst,f"burst_{batch_idx}_{cfg.n2n}.png")

        # -- crop images --
        if True: #cfg.abps or cfg.abps_inputs:
            images = [burst,sim_burst,raw_img,raw_img_iid]
            spacing = burst.shape[0] # we use frames as spacing
            cropped = crop_center_patch(images,spacing,cfg.frame_size)
            burst,sim_burst = cropped[0],cropped[1]
            raw_img,raw_img_iid = cropped[2],cropped[3]
            burst = burst[:cfg.nframes] # last frame is target
            if cfg.abps or cfg.abps_inputs:
                aligned = crop_center_patch([aligned],spacing,cfg.frame_size)[0]

        # -- getting shapes of data --
        N,B,C,H,W = burst.shape
        burst_og = burst.clone()

        # -- shuffle over Simulated Samples --
        k_ins,k_outs = create_k_grid(sim_burst,shuffle=True)
        k_ins,k_outs = [k_ins[0]],[k_outs[0]]
        # k_ins,k_outs = create_k_grid_v3(sim_burst)

        for k_in,k_out in zip(k_ins,k_outs):
            if k_in == k_out: continue

            # -- zero gradients; ready 2 go --
            model.align_info.model.zero_grad()
            model.align_info.optim.zero_grad()
            model.denoiser_info.model.zero_grad()
            model.denoiser_info.optim.zero_grad()
            model.unet_info.model.zero_grad()
            model.unet_info.optim.zero_grad()

            # -- compute input/output data --
            if cfg.sim_only_middle and (not cfg.abps):
                # sim_burst.shape == T,B,K,C,H,W
                midi = 0 if sim_burst.shape[0] == 1 else N//2
                left_burst,right_burst = burst[:N//2],burst[N//2+1:]
                burst = torch.cat([left_burst,sim_burst[[midi],:,k_in],right_burst],dim=0)
                mid_img = sim_burst[midi,:,k_out]
            elif cfg.abps and (not cfg.abps_inputs):
                # -- v1 --
                mid_img = aligned[-1]

                # -- v2 --
                # left_aligned,right_aligned = aligned[:N//2],aligned[N//2+1:]
                # nc_aligned = torch.cat([left_aligned,right_aligned],dim=0)
                # shuf = shuffle_aligned_pixels(nc_aligned,cfg.nframes)
                # mid_img = shuf[1]


                # ---- v3 ----
                # shuf = shuffle_aligned_pixels(aligned)
                # shuf = aligned[[N//2,0]]
                # midi = 0 if sim_burst.shape[0] == 1 else N//2
                # left_burst,right_burst = burst[:N//2],burst[N//2+1:]
                # burst = torch.cat([left_burst,shuf[[0]],right_burst],dim=0)
                # nc_burst = torch.cat([left_burst,right_burst],dim=0)
                # shuf = shuffle_aligned_pixels(aligned)

                # ---- v4 ----
                # nc_shuf = shuffle_aligned_pixels(nc_aligned)
                # mid_img = nc_shuf[0]
                # pick = npr.randint(0,2,size=(1,))[0]
                # mid_img = nc_aligned[pick]
                # mid_img = shuf[1]

                # save_image(shuf,"shuf.png")
                # print(shuf.shape)
                
                # diff = raw_img.cuda(non_blocking=True) - aligned[0]
                # mean = torch.mean(diff).item()
                # std = torch.std(diff).item()
                # print(mean,std)

                # -- v1 --
                # burst = burst
                # notMid = sample_not_mid(N)
                # mid_img = aligned[notMid]

            elif cfg.abps_inputs:
                burst = aligned.clone()
                burst_og = aligned.clone()
                mid_img = shuffle_aligned_pixels(burst,cfg.nframes)[0]

            else:
                burst = sim_burst[:,:,k_in]
                mid_img = sim_burst[N//2,:,k_out]
            # mid_img =  sim_burst[N//2,:]
            # print(burst.shape,mid_img.shape)
            # print(F.mse_loss(burst,mid_img).item())
            if cfg.supervised: gt_img = get_nmlz_tgt_img(cfg,raw_img).cuda(non_blocking=True)
            elif cfg.n2n: gt_img = raw_img_iid #noise_xform(raw_img).cuda(non_blocking=True)
            else: gt_img = mid_img
            

            # for bt in range(cfg.nframes):
            #     tiled = tile_across_blocks(burst[[bt]],cfg.nblocks)
            #     rep_burst = repeat(burst,'t b c h w -> t b g c h w',g=tiled.shape[2])
            #     for t in range(cfg.nframes):
            #         save_image(tiled[0] - rep_burst[t],f"tiled_{bt}_sub_burst_{t}.png")
            #         print_tensor_stats(f"delta_{bt}_{t}",tiled[0,:,4] - burst[t])

            # raw_img = raw_img.cuda(non_blocking=True) - 0.5
            # print_tensor_stats("gt_img - raw",gt_img - raw_img)
            # # save_image(gt_img,"gt.png")
            # # save_image(raw,"raw.png")
            # save_image(gt_img - raw_img,"gt_sub_raw.png")
            # print_tensor_stats("burst[N//2] - raw",burst[N//2] - raw_img)
            # save_image(burst[N//2] - raw_img,"burst_sub_raw.png")
            # print_tensor_stats("burst[N//2] - gt_img",burst[N//2] - gt_img)
            # save_image(burst[N//2] - gt_img,"burst_sub_gt.png")
            # print_tensor_stats("aligned[N//2] - raw",aligned[N//2] - raw_img)
            # save_image(aligned[N//2] - raw_img,"aligned_sub_raw.png")
            # print_tensor_stats("aligned[N//2] - burst[N//2]",
            # aligned[N//2] - burst[N//2])
            # save_image(aligned[N//2] - burst[N//2],"aligned_sub_burst.png")
            # gt_img = torch.normal(raw_zm_img,noise_level/255.)
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #        Dataset Augmentation
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            
            # burst,gt_img = apply_transformations(burst,gt_img)

            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #      Formatting Images for FP
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            stacked_burst = rearrange(burst,'n b c h w -> b n c h w')
            cat_burst = rearrange(burst,'n b c h w -> (b n) c h w')

            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #           Foward Pass
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            outputs = model(burst)
            m_aligned,m_aligned_ave,denoised,denoised_ave = outputs[:4]
            aligned_filters,denoised_filters = outputs[4:]
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            # 
            #    Decrease Entropy within a Kernel
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            filters_entropy = 0
            filters_entropy_coeff = 0. # 1000.
            all_filters = []
            L = len(align_hook.filters)
            iter_filters = align_hook.filters if L > 0 else [aligned_filters]
            for filters in iter_filters:
                f_shape = 'b n k2 c h w -> (b n c h w) k2'
                filters_shaped = rearrange(filters,f_shape)
                filters_entropy += one #entropyLoss(filters_shaped)
                all_filters.append(filters)
            if L > 0: filters_entropy /= L 
            all_filters = torch.stack(all_filters,dim=1)
            align_hook.clear()
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #   Reconstruction Losses (MSE)
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            losses = [F.mse_loss(denoised_ave,gt_img)]
            # losses = denoiseLossMSE(denoised,denoised_ave,gt_img,cfg.global_step)
            # losses = [ one, one ]
            # ave_loss,burst_loss = [loss.item() for loss in losses]
            rec_mse = np.sum(losses)
            # rec_mse = F.mse_loss(denoised_ave,gt_img)
            rec_mse_coeff = 1.
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #    Reconstruction Losses (Distribution)
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            gt_img_rep = gt_img.unsqueeze(1).repeat(1,denoised.shape[1],1,1,1)
            residuals = denoised - gt_img_rep
            rec_ot = torch.FloatTensor([0.]).to(cfg.device)
            # rec_ot = kl_gaussian_bp(residuals,noise_level,flip=True)
            # rec_ot = kl_gaussian_bp_patches(residuals,noise_level,flip=True,patchsize=16)
            if torch.any(torch.isnan(rec_ot)): rec_ot = torch.FloatTensor([0.]).to(cfg.device)
            if torch.any(torch.isinf(rec_ot)): rec_ot = torch.FloatTensor([0.]).to(cfg.device)
            rec_ot_coeff = 0.
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #              Final Losses
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            rec_loss = rec_mse_coeff * rec_mse + rec_ot_coeff * rec_ot
            final_loss = rec_loss
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #              Record Keeping
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            # -- reconstruction MSE --
            rec_mse_losses += rec_mse.item()
            rec_mse_count += 1
    
            # -- reconstruction Dist. --
            rec_ot_losses += rec_ot.item()
            rec_ot_count += 1
    
            # -- total loss --
            running_loss += final_loss.item()
            total_loss += final_loss.item()
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #        Gradients & Backpropogration
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            # -- compute the gradients! --
            if cfg.use_seed: torch.set_deterministic(False)
            final_loss.backward()
            if cfg.use_seed: torch.set_deterministic(True)
    
            # -- backprop now. --
            model.align_info.optim.step()
            model.denoiser_info.optim.step()
            model.unet_info.optim.step()
            scheduler.step()

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #            Printing to Stdout
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:


            # -- recompute model output for original images --
            outputs = model(burst_og)
            m_aligned,m_aligned_ave,denoised,denoised_ave = outputs[:4]
            aligned_filters,denoised_filters = outputs[4:]

            # -- compute mse for fun --
            B = raw_img.shape[0]            
            raw_img = raw_img.cuda(non_blocking=True)
            raw_img = get_nmlz_tgt_img(cfg,raw_img)

            # -- psnr for [average of aligned frames] --
            mse_loss = F.mse_loss(raw_img,m_aligned_ave,reduction='none').reshape(B,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_aligned_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_aligned_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [average of input, misaligned frames] --
            mis_ave = torch.mean(burst_og,dim=0)
            if noise_type == "qis": mis_ave = quantize_img(cfg,mis_ave)
            mse_loss = F.mse_loss(raw_img,mis_ave,reduction='none').reshape(B,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_misaligned_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_misaligned_std = np.std(mse_to_psnr(mse_loss))

            # tv_utils.save_image(raw_img,"raw.png",nrow=1,normalize=True,range=(-0.5,1.25))
            # tv_utils.save_image(mis_ave,"mis.png",nrow=1,normalize=True,range=(-0.5,1.25))

            # -- psnr for [bm3d] --
            mid_img_og = burst[N//2]
            bm3d_nb_psnrs = []
            M = 4 if B > 4 else B
            for b in range(M):
                bm3d_rec = bm3d.bm3d(mid_img_og[b].cpu().transpose(0,2)+0.5,
                                     sigma_psd=noise_level/255,
                                     stage_arg=bm3d.BM3DStages.ALL_STAGES)
                bm3d_rec = torch.FloatTensor(bm3d_rec).transpose(0,2)
                # maybe an issue here
                b_loss = F.mse_loss(raw_img[b].cpu(),bm3d_rec,reduction='none').reshape(1,-1)
                b_loss = torch.mean(b_loss,1).detach().cpu().numpy()
                bm3d_nb_psnr = np.mean(mse_to_psnr(b_loss))
                bm3d_nb_psnrs.append(bm3d_nb_psnr)
            bm3d_nb_ave = np.mean(bm3d_nb_psnrs)
            bm3d_nb_std = np.std(bm3d_nb_psnrs)

            # -- psnr for input averaged frames --
            # burst_ave = torch.mean(burst_og,dim=0)
            # mse_loss = F.mse_loss(raw_img,burst_ave,reduction='none').reshape(B,-1)
            # mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            # psnr_input_ave = np.mean(mse_to_psnr(mse_loss))
            # psnr_input_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for aligned + denoised --
            R = denoised.shape[1]
            raw_img_repN = raw_img.unsqueeze(1).repeat(1,R,1,1,1)
            # if noise_type == "qis": denoised = quantize_img(cfg,denoised)
            # save_image(denoised_ave,"denoised_ave.png")
            # save_image(denoised,"denoised.png")
            mse_loss = F.mse_loss(raw_img_repN,denoised,reduction='none').reshape(B,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_denoised_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_denoised_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [model output image] --
            mse_loss = F.mse_loss(raw_img,denoised_ave,reduction='none').reshape(B,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr = np.mean(mse_to_psnr(mse_loss))
            psnr_std = np.std(mse_to_psnr(mse_loss))

            # -- update losses --
            running_loss /= cfg.log_interval

            # -- reconstruction MSE --
            rec_mse_ave = rec_mse_losses / rec_mse_count
            rec_mse_losses,rec_mse_count = 0,0

            # -- reconstruction Dist. --
            rec_ot_ave = rec_ot_losses / rec_ot_count 
            rec_ot_losses,rec_ot_count = 0,0

            # -- write record --
            if use_record:
                info = {'burst':burst_loss,'ave':ave_loss,'ot':rec_ot_ave,
                        'psnr':psnr,'psnr_std':psnr_std}
                record_losses = record_losses.append(info,ignore_index=True)
                
            # -- write to stdout --
            write_info = (epoch, cfg.epochs, batch_idx, steps_per_epoch,running_loss,
                          psnr,psnr_std,psnr_denoised_ave,psnr_denoised_std,psnr_aligned_ave,
                          psnr_aligned_std,psnr_misaligned_ave,psnr_misaligned_std,bm3d_nb_ave,
                          bm3d_nb_std,rec_mse_ave,rec_ot_ave)
            print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.2f +/- %2.2f [den]: %2.2f +/- %2.2f [al]: %2.2f +/- %2.2f [mis]: %2.2f +/- %2.2f [bm3d]: %2.2f +/- %2.2f [r-mse]: %.2e [r-ot]: %.2e" % write_info)
            # -- write to summary writer --
            if writer:
                writer.add_scalar('train/running-loss',running_loss,cfg.global_step)
                writer.add_scalars('train/model-psnr',{'ave':psnr,'std':psnr_std},cfg.global_step)
                writer.add_scalars('train/dn-frame-psnr',{'ave':psnr_denoised_ave,
                                                          'std':psnr_denoised_std},cfg.global_step)

            # -- reset loss --
            running_loss = 0

        # -- write examples --
        if write_examples and (batch_idx % write_examples_iter) == 0 and (batch_idx > 0 or cfg.global_step == 0):
            write_input_output(cfg,model,stacked_burst,aligned,denoised,all_filters,motion)

        if use_timer: clock.toc()

        if use_timer:
            print("data_clock",data_clock.average_time)
            print("clock",clock.average_time)
        cfg.global_step += 1

    # -- remove hooks --
    for hook in align_hooks: hook.remove()

    total_loss /= len(train_loader)
    return total_loss,record_losses

def add_color_channel(bw_pic):
    repeat = [1 for i in bw_pic.shape]
    repeat[-3] = 3
    bw_pic = bw_pic.repeat(*(repeat))
    return bw_pic

def quantize_img(cfg,image):
    params = cfg.noise_params['qis']
    pix_max = 2**params['nbits'] - 1
    image += 0.5
    image *= params['alpha']
    image = torch.round(image)
    image = torch.clamp(image, 0, pix_max)
    image /= params['alpha']
    image -= 0.5
    return image
    
def get_nmlz_tgt_img(cfg,raw_img):
    pix_max = 2**3-1
    noise_type = cfg.noise_params.ntype
    if noise_type in ["g","hg"]: nmlz_raw = raw_img - 0.5
    elif noise_type in ["qis"]:
        params = cfg.noise_params[noise_type]
        pix_max = 2**params['nbits'] - 1
        raw_img_bw = tvF.rgb_to_grayscale(raw_img,1)
        raw_img_bw = add_color_channel(raw_img_bw)
        # nmlz_raw = raw_scale * raw_img_bw - 0.5
        # raw_img_bw *= params['alpha']
        # raw_img_bw = torch.round(raw_img_bw)
        # print("ll",ll_pic.min().item(),ll_pic.max().item())
        # raw_img_bw = torch.clamp(raw_img_bw, 0, pix_max)
        # raw_img_bw /= params['alpha']
        # -- end of qis noise transform --

        # -- start dnn normalization for optimization --
        nmlz_raw = raw_img_bw - 0.5
    else:
        print("[Warning]: Check normalize raw image.")        
        nmlz_raw = raw_img
    return nmlz_raw

def test_loop(cfg,model,test_loader,epoch):
    model.eval()
    model.align_info.model.eval()
    model.denoiser_info.model.eval()
    model.unet_info.model.eval()
    model = model.to(cfg.device)
    noise_type = cfg.noise_params.ntype
    total_psnr = 0
    total_loss = 0
    use_record = False
    record_test = pd.DataFrame({'psnr':[]})

    init = torch.initial_seed()
    torch.manual_seed(cfg.seed+1+epoch+init)
    test_iter = iter(test_loader)
    num_batches,D = 25,len(test_iter) 
    num_batches = D
    num_batches = num_batches if D > num_batches else D
    psnrs = np.zeros( ( num_batches, cfg.batch_size ) )

    with torch.no_grad():
        for batch_idx in range(num_batches):

            sample = next(test_iter)
            burst,raw_img,motion = sample['burst'],sample['clean'],sample['directions']
            B = raw_img.shape[0]
            
            # -- selecting input frames --
            input_order = np.arange(cfg.N)
            # print("pre",input_order)
            middle_img_idx = -1
            if not cfg.input_with_middle_frame:
                middle = cfg.N // 2
                # print(middle)
                middle_img_idx = input_order[middle]
                # input_order = np.r_[input_order[:middle],input_order[middle+1:]]
            else:
                middle = len(input_order) // 2
                input_order = np.arange(cfg.N)
                middle_img_idx = input_order[middle]
                # input_order = np.arange(cfg.N)
            
            # -- reshaping of data --
            raw_img = raw_img.cuda(non_blocking=True)
            burst = burst.cuda(non_blocking=True)
            stacked_burst = torch.stack([burst[input_order[x]] for x in range(cfg.input_N)],dim=1)
            cat_burst = torch.cat([burst[input_order[x]] for x in range(cfg.input_N)],dim=1)
    

            # -- align images if necessary --
            if cfg.abps_inputs:
                # scores,aligned = abp_search(cfg,burst)
                # scores,aligned = lpas_search(cfg,burst,motion)
                mtype = "global"
                acc = cfg.optical_flow_acc
                scores,aligned = lpas_spoof(burst,motion,cfg.nblocks,mtype,acc)
                burst = aligned.clone()

            if True:
                images = [burst,raw_img]
                cropped = crop_center_patch(images,cfg.nframes,cfg.frame_size)
                burst,raw_img = cropped[0],cropped[1]
                if cfg.abps_inputs:
                    aligned = crop_center_patch([aligned],spacing,cfg.frame_size)[0]
                burst = burst[:cfg.nframes]


            # -- denoising --
            m_aligned,m_aligned_ave,denoised,denoised_ave,a_filters,d_filters = model(burst)
            denoised_ave = denoised_ave.detach()

            # if not cfg.input_with_middle_frame:
            #     denoised_ave = model(cat_burst,stacked_burst)[1]
            # else:
            #     denoised_ave = model(cat_burst,stacked_burst)[0][middle_img_idx]

            # denoised_ave = burst[middle_img_idx] - rec_res
            
            # -- compare with stacked targets --
            raw_img = get_nmlz_tgt_img(cfg,raw_img)
            # denoised_ave = rescale_noisy_image(denoised_ave)        

            # -- compute psnr --
            loss = F.mse_loss(raw_img,denoised_ave,reduction='none').reshape(B,-1)
            # loss = F.mse_loss(raw_img,burst[cfg.input_N//2]+0.5,reduction='none').reshape(B,-1)
            loss = torch.mean(loss,1).detach().cpu().numpy()
            psnr = mse_to_psnr(loss)
            psnrs[batch_idx,:] = psnr
                        
            if use_record:
                record_test = record_test.append({'psnr':psnr},ignore_index=True)
            total_psnr += np.mean(psnr)
            total_loss += np.mean(loss)

            # if (batch_idx % cfg.test_log_interval) == 0:
            #     root = Path(f"{settings.ROOT_PATH}/output/n2n/offset_out_noise/denoised_aves/N{cfg.N}/e{epoch}")
            #     if not root.exists(): root.mkdir(parents=True)
            #     fn = root / Path(f"b{batch_idx}.png")
            #     nrow = int(np.sqrt(cfg.batch_size))
            #     denoised_ave = denoised_ave.detach().cpu()
            #     grid_imgs = tv_utils.make_grid(denoised_ave, padding=2, normalize=True, nrow=nrow)
            #     plt.imshow(grid_imgs.permute(1,2,0))
            #     plt.savefig(fn)
            #     plt.close('all')
            if batch_idx % 100 == 0: print("[%d/%d] Test PSNR: %2.2f" % (batch_idx,num_batches,total_psnr / (batch_idx+1)))

    psnr_ave = np.mean(psnrs)
    psnr_std = np.std(psnrs)
    ave_loss = total_loss / num_batches
    print("[N: %d] Testing: [psnr: %2.2f +/- %2.2f] [ave loss %2.3e]"%(cfg.N,psnr_ave,psnr_std,ave_loss))
    return psnr_ave,record_test


def write_input_output(cfg,model,burst,aligned,denoised,filters,motion):

    """
    :params burst: input images to the model, :shape [B, N, C, H, W]
    :params aligned: output images from the alignment layers, :shape [B, N, C, H, W]
    :params denoised: output images from the denoiser, :shape [B, N, C, H, W]
    :params filters: filters used by model, :shape [B, L, N, K2, 1, Hf, Wf] with Hf = (H or 1) for L = number of cascaded filters
    """

    # -- file path --
    path = Path(f"./output/n2sim/io_examples/{cfg.exp_name}/")
    if not path.exists(): path.mkdir(parents=True)

    # -- init --
    B,N,C,H,W = burst.shape

    # -- save histogram of residuals --
    denoised_np = denoised.detach().cpu().numpy()
    plot_histogram_residuals_batch(denoised_np,cfg.global_step,path,rand_name=False)

    # -- save histogram of gradients (denoiser) --
    if not model.use_unet_only:
        denoiser = model.denoiser_info.model
        plot_histogram_gradients(denoiser,"denoiser",cfg.global_step,path,rand_name=False)

    # -- save histogram of gradients (alignment) --
    if model.use_alignment:
        alignment = model.align_info.model
        plot_histogram_gradients(alignment,"alignment",cfg.global_step,path,rand_name=False)

    # -- save gradient norm by layer (denoiser) --
    if not model.use_unet_only:
        denoiser = model.denoiser_info.model
        plot_histogram_gradient_norms(denoiser,"denoiser",cfg.global_step,path,rand_name=False)

    # -- save gradient norm by layer (alignment) --
    if model.use_alignment:
        alignment = model.align_info.model
        plot_histogram_gradient_norms(alignment,"alignment",cfg.global_step,path,rand_name=False)

    if B > 4: B = 4
    for b in range(B):
        
        # -- save dirty & clean & res triplet --
        fn = path / Path(f"image_{cfg.global_step}_{b}.png")
        res = burst[b][N//2] - denoised[b].mean(0)
        imgs = torch.stack([burst[b][N//2],denoised[b].mean(0),res],dim=0)
        tv_utils.save_image(imgs,fn,nrow=3,normalize=True,range=(-0.5,0.5))
        
        # -- save images --
        fn = path / Path(f"{cfg.global_step}_{b}.png")
        burst_b = torch.cat([burst[b][[N//2]] - burst[b][[0]],burst[b],burst[b][[N//2]] - burst[b][[-1]]],dim=0)
        aligned_b = torch.cat([aligned[b][[N//2]] - aligned[b][[0]],aligned[b],aligned[b][[N//2]] - aligned[b][[-1]]],dim=0)
        denoised_b = torch.cat([denoised[b][[N//2]] - denoised[b][[0]],denoised[b],denoised[b][[N//2]] - denoised[b][[-1]]],dim=0)
        imgs = torch.cat([burst_b,aligned_b,denoised_b],dim=0) # 2N,C,H,W
        tv_utils.save_image(imgs,fn,nrow=N+2,normalize=True,range=(-0.5,0.5))

        # -- save filters --
        fn = path / Path(f"filters_{cfg.global_step}_{b}.png")
        K = int(np.sqrt(filters.shape[3]))
        L = filters.shape[1]
        if filters.shape[-1] > 1:
            S = npr.permutation(filters.shape[-1])[:10]
            filters_b = filters[b,...,0,S,S].view(N*10*L,1,K,K)
        else: filters_b = filters[b,...,0,0,0].view(N*L,1,K,K)
        tv_utils.save_image(filters_b,fn,nrow=N,normalize=True)

        # -- save direction image --
        fn = path / Path(f"arrows_{cfg.global_step}_{b}.png")
        if len(motion[b]) > 1 and len(motion[b].shape) > 1:
            arrows = create_arrow_image(motion[b],pad=2)
            tv_utils.save_image([arrows],fn)


    print(f"Wrote example images to file at [{path}]")
    plt.close("all")



def create_arrow_image(motion,pad=2):
    D = len(motion)
    assert D == 1,f"Only one direction right now. Currently, [{len(motion)}]"
    W = 100
    S = (W + pad) * D + pad
    arrows = np.zeros((S,W+2*pad,3))
    direction = motion[0]
    for i in range(D):
        col_i = (pad+W)*i+pad
        canvas = arrows[col_i:col_i+W,pad:pad+W,:]
        start_point = (0,0)
        x_end = direction[0].item()
        y_end = direction[1].item()
        end_point = (x_end,y_end)

        fig = Figure(dpi=300)
        plt_canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.annotate("",
                    xy=end_point, xycoords='data',
                    xytext=start_point, textcoords='data',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),
        )
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        plt_canvas.draw()       # draw the canvas, cache the renderer
        canvas = np.array(plt_canvas.buffer_rgba())[:,:,:]
        arrows = canvas
    arrows = torch.Tensor(arrows.astype(np.uint8)).transpose(0,2).transpose(1,2)
    return arrows


