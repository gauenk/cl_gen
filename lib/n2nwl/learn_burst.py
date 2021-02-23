
# -- python imports --
import bm3d
import pandas as pd
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange, repeat, reduce
from easydict import EasyDict as edict

# -- just to draw an fing arrow --
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms.functional as tvF

# -- project code --
import settings
from pyutils.timer import Timer
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transform import ScaleZeroMean
from layers.ot_pytorch import sink_stabilized,sink,pairwise_distances
from layers.burst import BurstRecLoss,EntropyLoss

# -- [local] project code --
from .dist_loss import ot_pairwise_bp,ot_gaussian_bp,ot_pairwise2gaussian_bp,kl_gaussian_bp,w_gaussian_bp,kl_gaussian_bp_patches,kl_gaussian_pair_bp,w_gaussian_bp_patches
from .plot import plot_histogram_residuals_batch,plot_histogram_gradients,plot_histogram_gradient_norms
from .misc import AlignmentFilterHooks

def train_loop(cfg,model,train_loader,epoch,record_losses):


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

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Init Record Keeping
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    align_mse_losses,align_mse_count = 0,0
    align_ot_losses,align_ot_count = 0,0
    rec_mse_losses,rec_mse_count = 0,0
    rec_ot_losses,rec_ot_count = 0,0
    running_loss,total_loss = 0,0

    write_examples = True
    noise_level = cfg.noise_params['g']['stddev']

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

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Init Loss Functions
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    alignmentLossMSE = BurstRecLoss()
    denoiseLossMSE = BurstRecLoss()
    # denoiseLossOT = BurstResidualLoss()
    entropyLoss = EntropyLoss()

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #    Final Configs
    #
    # -=-=-=-=-=-=-=-=-=-=-

    use_timer = False
    one = torch.FloatTensor([1.]).to(cfg.device)
    switch = True
    if use_timer: clock = Timer()
    train_iter = iter(train_loader)
    steps_per_epoch = len(train_loader)
    write_examples_iter = steps_per_epoch//2

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #     Start Epoch
    #
    # -=-=-=-=-=-=-=-=-=-=-

    for batch_idx in range(steps_per_epoch):

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #      Setting up for Iteration
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- setup iteration timer --
        if use_timer: clock.tic()

        # -- zero gradients; ready 2 go --
        model.align_info.model.zero_grad()
        model.align_info.optim.zero_grad()
        model.denoiser_info.model.zero_grad()
        model.denoiser_info.optim.zero_grad()
        model.unet_info.model.zero_grad()
        model.unet_info.optim.zero_grad()


        # -- grab data batch --
        burst, res_imgs, raw_img, directions = next(train_iter)

        # -- getting shapes of data --
        N,B,C,H,W = burst.shape
        burst = burst.cuda(non_blocking=True)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #      Formatting Images for FP
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- creating some transforms --
        stacked_burst = rearrange(burst,'n b c h w -> b n c h w')
        cat_burst = rearrange(burst,'n b c h w -> (b n) c h w')

        # -- extract target image --
        mid_img =  burst[N//2]
        raw_zm_img = szm(raw_img.cuda(non_blocking=True))
        if cfg.supervised: gt_img = raw_zm_img
        else: gt_img = mid_img


        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #              Check Some Gradients
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


        def mse_v_wassersteinG_check_some_gradients(cfg,burst,gt_img,model):
            grads = edict()
            gt_img_rs = gt_img.unsqueeze(1).repeat(1,N,1,1,1)
            model.unet_info.model.zero_grad()
            burst.requires_grad_(True)

            outputs = model(burst)
            aligned,aligned_ave,denoised,denoised_ave = outputs[:4]
            aligned_filters,denoised_filters = outputs[4:]
            residuals = denoised - gt_img_rs
            P = 1.#residuals.numel()
            denoised.retain_grad()
            rec_mse = (denoised.reshape(B,-1) - gt_img.reshape(B,-1))**2
            rec_mse.retain_grad()
            ones = P*torch.ones_like(rec_mse)
            rec_mse.backward(ones,retain_graph=True)
            grads.rmse = rec_mse.grad.clone().reshape(B,-1)
            grad_rec_mse = grads.rmse
            grads.dmse = denoised.grad.clone().reshape(B,-1)
            grad_denoised_mse = grads.dmse
            ones = torch.ones_like(rec_mse)
            grads.d_to_b = torch.autograd.grad(rec_mse,denoised,ones)[0].reshape(B,-1)

            model.unet_info.model.zero_grad()
            outputs = model(burst)
            aligned,aligned_ave,denoised,denoised_ave = outputs[:4]
            aligned_filters,denoised_filters = outputs[4:]
            # residuals = denoised - gt_img_rs
            # rec_ot = w_gaussian_bp(residuals,noise_level)
            denoised.retain_grad()
            rec_ot_v =  (denoised-gt_img_rs)**2 
            rec_ot_v.retain_grad()
            rec_ot = (rec_ot_v.mean() - noise_level/255.)**2
            rec_ot.retain_grad()
            ones = P*torch.ones_like(rec_ot)
            rec_ot.backward(ones)
            grad_denoised_ot = denoised.grad.clone().reshape(B,-1)
            grads.dot = grad_denoised_ot
            grad_rec_ot = rec_ot_v.grad.clone().reshape(B,-1)
            grads.rot = grad_denoised_ot

            print("Gradient Name Info")
            for name,g in grads.items():
                g_norm = g.norm().item()
                g_mean = g.mean().item()
                g_std = g.std().item()
                print(name,g.shape,g_norm,g_mean,g_std)

            print_pairs = False
            if print_pairs:
                print("All Gradient Ratios")
                for name_t,g_t in grads.items():
                    for name_b,g_b in grads.items():
                        ratio = g_t/g_b
                        ratio_m = ratio.mean().item()
                        ratio_std = ratio.std().item()
                        print("[%s/%s] [%2.2e +/- %2.2e]" % (name_t,name_b,ratio_m,ratio_std))

            use_true_mse = False
            if use_true_mse:
                print("Ratios with Estimated MSE Gradient")
                true_dmse = 2*torch.mean(denoised_ave-gt_img)**2
                ratio_mse = grads.dmse/true_dmse
                ratio_mse_dtb = grads.dmse/grads.d_to_b
                print(ratio_mse)
                print(ratio_mse_dtb)
            
            dot_v_dmse = True
            if dot_v_dmse:
                print("Ratio of Denoised OT and Denoised MSE")
                ratio_mseot = (grads.dmse / grads.dot)
                print(ratio_mseot.mean(),ratio_mseot.std())
                ratio_mseot = ratio_mseot[0,0].item()

                c1 = torch.mean( (denoised-gt_img_rs)**2 ).item()
                c2 = noise_level/255
                m = torch.mean( gt_img_rs ).item()
                true_ratio = 2.*(c1 - c2) / ( np.product(burst.shape) )
                # diff = denoised.reshape(B,-1)-gt_img_rs.reshape(B,-1)
                # true_ratio = 2.*(c1 - c2) * ( diff / ( np.product(burst.shape) ) )
                # print(c1,c2,m,true_ratio,1./true_ratio)
                ratio_mseot = (grads.dmse / ( grads.dot ) )
                print(ratio_mseot*true_ratio)

                # ratio_mseot = (grads.dmse / ( grads.dot / diff) )
                # print(ratio_mseot*true_ratio)
                # print(ratio_mseot.mean(),ratio_mseot.std())

            exit()
            model.unet_info.model.zero_grad()

        # mse_v_wassersteinG_check_some_gradients(cfg,burst,gt_img,model)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #           Foward Pass
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        outputs = model(burst)
        aligned,aligned_ave,denoised,denoised_ave = outputs[:4]
        aligned_filters,denoised_filters = outputs[4:]
        
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # 
        #   Require Approx Equal Filter Norms (aligned)
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        aligned_filters_rs = rearrange(aligned_filters,'b n k2 c h w -> b n (k2 c h w)')
        norms = torch.norm(aligned_filters_rs,p=2.,dim=2)
        norms_mid = norms[:,N//2].unsqueeze(1).repeat(1,N)
        norm_loss_align = torch.mean( torch.pow( torch.abs(norms - norms_mid), 1. ) )

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # 
        #   Require Approx Equal Filter Norms (denoised)
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        denoised_filters = rearrange(denoised_filters,'b n k2 c h w -> b n (k2 c h w)')
        norms = torch.norm(denoised_filters,p=2.,dim=2)
        norms_mid = norms[:,N//2].unsqueeze(1).repeat(1,N)
        norm_loss_denoiser = torch.mean( torch.pow( torch.abs(norms - norms_mid), 1. ) )
        norm_loss_coeff = 0.

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
            filters_shaped = rearrange(filters,'b n k2 c h w -> (b n c h w) k2',n=N)
            filters_entropy += entropyLoss(filters_shaped)
            all_filters.append(filters)
        if L > 0: filters_entropy /= L 
        all_filters = torch.stack(all_filters,dim=1)
        align_hook.clear()

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #    Increase Entropy across each Kernel
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        filters_dist_entropy = 0

        # -- across each frame --
        # filters_shaped = rearrange(all_filters,'b l n k2 c h w -> (b l) (n c h w) k2')
        # filters_shaped = torch.mean(filters_shaped,dim=1)
        # filters_dist_entropy += -1 * entropyLoss(filters_shaped)

        # -- across each batch --
        filters_shaped = rearrange(all_filters,'b l n k2 c h w -> (n l) (b c h w) k2')
        filters_shaped = torch.mean(filters_shaped,dim=1)
        filters_dist_entropy += -1 * entropyLoss(filters_shaped)

        # -- across each kpn cascade --
        # filters_shaped = rearrange(all_filters,'b l n k2 c h w -> (b n) (l c h w) k2')
        # filters_shaped = torch.mean(filters_shaped,dim=1)
        # filters_dist_entropy += -1 * entropyLoss(filters_shaped)

        filters_dist_coeff = 0

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #    Alignment Losses (MSE)
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        losses = alignmentLossMSE(aligned,aligned_ave,gt_img,cfg.global_step)
        ave_loss,burst_loss = [loss.item() for loss in losses]
        align_mse = np.sum(losses)
        align_mse_coeff = 0. #0.95**cfg.global_step


        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #   Alignment Losses (Distribution)
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # pad = 2*cfg.N
        # fs = cfg.dynamic.frame_size
        residuals = aligned - gt_img.unsqueeze(1).repeat(1,N,1,1,1)
        # centered_residuals = tvF.center_crop(residuals,(fs-pad,fs-pad))
        # centered_residuals = tvF.center_crop(residuals,(fs//2,fs//2))
        # align_ot = kl_gaussian_bp(residuals,noise_level,flip=True)
        align_ot = kl_gaussian_bp_patches(residuals,noise_level,flip=True,patchsize=16)
        align_ot_coeff = 0# 100.

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #   Reconstruction Losses (MSE)
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        losses = denoiseLossMSE(denoised,denoised_ave,gt_img,cfg.global_step)
        ave_loss,burst_loss = [loss.item() for loss in losses]
        rec_mse = np.sum(losses)
        rec_mse_coeff = 0. #0.9**cfg.global_step

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #    Reconstruction Losses (Distribution)
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- computation --
        gt_img_rs = gt_img.unsqueeze(1).repeat(1,N,1,1,1)
        residuals = denoised - gt_img.unsqueeze(1).repeat(1,N,1,1,1)
        # rec_ot_a = kl_gaussian_bp(residuals,noise_level)
        # rec_ot_b = kl_gaussian_bp(residuals,noise_level,flip=True)
        # rec_ot = (rec_ot_a + rec_ot_b)/2.
        # alpha_grid = [0.,1.,5.,10.,25.]
        # for alpha in alpha_grid:
        #     # residuals = torch.normal(torch.zeros_like(residuals)+ gt_img_rs*alpha/255.,noise_level/255.)
        #     residuals = torch.normal(torch.zeros_like(residuals),noise_level/255.+ gt_img_rs*alpha/255.)

        #     rec_ot_v2_a = kl_gaussian_bp_patches(residuals,noise_level,patchsize=16)
        #     rec_ot_v1_b = kl_gaussian_bp(residuals,noise_level,flip=True)
        #     rec_ot_v2_b = kl_gaussian_bp_patches(residuals,noise_level,flip=True,patchsize=16)
        #     rec_ot_all = torch.tensor([rec_ot_v1_a,rec_ot_v2_a,rec_ot_v1_b,rec_ot_v2_b])

        #     rec_ot_v2 = (rec_ot_v2_a + rec_ot_v2_b).item()/2.
        #     print(alpha,torch.min(rec_ot_all),torch.max(rec_ot_all),rec_ot_v1,rec_ot_v2)
        # exit()
        rec_ot = w_gaussian_bp(residuals,noise_level)
        # print(residuals.numel())
        rec_ot_coeff = 100. #residuals.numel()*2.
        # 1000.# - .997**cfg.global_step

        # residuals = rearrange(residuals,'b n c h w -> b n (h w) c')
        # rec_ot_pair_loss_v1 = w_gaussian_bp(residuals,noise_level)
        # rec_ot_loss_v1 = kl_gaussian_bp(residuals,noise_level,flip=True)
        # rec_ot_loss_v1 = kl_gaussian_pair_bp(residuals)
        # rec_ot_loss_v1 = ot_pairwise2gaussian_bp(residuals,K=6,reg=reg)
        # rec_ot_loss_v2 = ot_pairwise_bp(residuals,K=3)
        # rec_ot_pair_loss_v2 = torch.FloatTensor([0.]).to(cfg.device)
        # rec_ot = (rec_ot_loss_v1 + rec_ot_pair_loss_v2)
            
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #              Final Losses
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        rec_loss = rec_ot_coeff * rec_ot + rec_mse_coeff * rec_mse
        norm_loss = norm_loss_coeff * ( norm_loss_denoiser + norm_loss_align)
        align_loss = align_mse_coeff * align_mse + align_ot_coeff * align_ot
        entropy_loss = 0 #filters_entropy_coeff * filters_entropy + filters_dist_coeff * filters_dist_entropy 
        # final_loss = align_loss + rec_loss + entropy_loss + norm_loss
        final_loss = rec_loss

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #              Record Keeping
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- alignment MSE --
        align_mse_losses += align_mse.item()
        align_mse_count += 1

        # -- alignment Dist --
        align_ot_losses += align_ot.item()
        align_ot_count += 1

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
        final_loss.backward()

        # -- backprop now. --
        model.align_info.optim.step()
        model.denoiser_info.optim.step()
        model.unet_info.optim.step()

        # for name,params in model.unet_info.model.named_parameters():
        #     if not ("weight" in name): continue
        #     print(params.grad.norm())
        #     # print(module.conv1.parameters())
        #     # print(module.conv1.data.grad)


        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #            Printing to Stdout
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:


            # -- compute mse for fun --
            B = raw_img.shape[0]            
            raw_img = raw_img.cuda(non_blocking=True)

            # -- psnr for [average of aligned frames] --
            mse_loss = F.mse_loss(raw_img,aligned_ave+0.5,reduction='none').reshape(B,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_aligned_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_aligned_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [average of input, misaligned frames] --
            mis_ave = torch.mean(stacked_burst,dim=1)
            mse_loss = F.mse_loss(raw_img,mis_ave+0.5,reduction='none').reshape(B,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_misaligned_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_misaligned_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [bm3d] --
            bm3d_nb_psnrs = []
            M = 10 if B > 10 else B
            for b in range(B):
                bm3d_rec = bm3d.bm3d(mid_img[b].cpu().transpose(0,2)+0.5,
                                     sigma_psd=noise_level/255,
                                     stage_arg=bm3d.BM3DStages.ALL_STAGES)
                bm3d_rec = torch.FloatTensor(bm3d_rec).transpose(0,2)
                b_loss = F.mse_loss(raw_img[b].cpu(),bm3d_rec,reduction='none').reshape(1,-1)
                b_loss = torch.mean(b_loss,1).detach().cpu().numpy()
                bm3d_nb_psnr = np.mean(mse_to_psnr(b_loss))
                bm3d_nb_psnrs.append(bm3d_nb_psnr)
            bm3d_nb_ave = np.mean(bm3d_nb_psnrs)
            bm3d_nb_std = np.std(bm3d_nb_psnrs)

            # -- psnr for aligned + denoised --
            raw_img_repN = raw_img.unsqueeze(1).repeat(1,N,1,1,1)
            mse_loss = F.mse_loss(raw_img_repN,denoised+0.5,reduction='none').reshape(B,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_denoised_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_denoised_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [model output image] --
            mse_loss = F.mse_loss(raw_img,denoised_ave+0.5,reduction='none').reshape(B,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr = np.mean(mse_to_psnr(mse_loss))
            psnr_std = np.std(mse_to_psnr(mse_loss))

            # -- update losses --
            running_loss /= cfg.log_interval

            # -- alignment MSE --
            align_mse_ave = align_mse_losses / align_mse_count
            align_mse_losses,align_mse_count = 0,0

            # -- alignment Dist. --
            align_ot_ave = align_ot_losses / align_ot_count
            align_ot_losses,align_ot_count = 0,0

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
            write_info = (epoch, cfg.epochs, batch_idx,len(train_loader),running_loss,
                          psnr,psnr_std,psnr_denoised_ave,psnr_denoised_std,psnr_aligned_ave,
                          psnr_aligned_std,psnr_misaligned_ave,psnr_misaligned_std,bm3d_nb_ave,
                          bm3d_nb_std,rec_mse_ave,rec_ot_ave)
            print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.2f +/- %2.2f [den]: %2.2f +/- %2.2f [al]: %2.2f +/- %2.2f [mis]: %2.2f +/- %2.2f [bm3d]: %2.2f +/- %2.2f [r-mse]: %.2e [r-ot]: %.2e" % write_info)
            running_loss = 0

        # -- write examples --
        if write_examples and (batch_idx % write_examples_iter) == 0 and (batch_idx > 0 or cfg.global_step == 0):
            write_input_output(cfg,model,stacked_burst,aligned,denoised,all_filters,directions)

        if use_timer: clock.toc()
        if use_timer: print(clock)
        cfg.global_step += 1

    # -- remove hooks --
    for hook in align_hooks: hook.remove()

    total_loss /= len(train_loader)
    return total_loss,record_losses

def test_loop(cfg,model,test_loader,epoch):
    model.eval()
    model.align_info.model.eval()
    model.denoiser_info.model.eval()
    model.unet_info.model.eval()
    model = model.to(cfg.device)
    total_psnr = 0
    total_loss = 0
    psnrs = np.zeros( (len(test_loader),cfg.batch_size) )
    use_record = False
    record_test = pd.DataFrame({'psnr':[]})

    with torch.no_grad():
        for batch_idx, (burst, res_imgs, raw_img, directions) in enumerate(test_loader):
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
    
            # -- denoising --
            aligned,aligned_ave,denoised,denoised_ave,a_filters,d_filters = model(burst)
            denoised_ave = denoised_ave.detach()

            # if not cfg.input_with_middle_frame:
            #     denoised_ave = model(cat_burst,stacked_burst)[1]
            # else:
            #     denoised_ave = model(cat_burst,stacked_burst)[0][middle_img_idx]

            # denoised_ave = burst[middle_img_idx] - rec_res
            
            # -- compare with stacked targets --
            denoised_ave = rescale_noisy_image(denoised_ave)        

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
            if batch_idx % 100 == 0: print("[%d/%d] Test PSNR: %2.2f" % (batch_idx,len(test_loader),total_psnr / (batch_idx+1)))

    psnr_ave = np.mean(psnrs)
    psnr_std = np.std(psnrs)
    ave_loss = total_loss / len(test_loader)
    print("[N: %d] Testing: [psnr: %2.2f +/- %2.2f] [ave loss %2.3e]"%(cfg.N,psnr_ave,psnr_std,ave_loss))
    return psnr_ave,record_test


def write_input_output(cfg,model,burst,aligned,denoised,filters,directions):

    """
    :params burst: input images to the model, :shape [B, N, C, H, W]
    :params aligned: output images from the alignment layers, :shape [B, N, C, H, W]
    :params denoised: output images from the denoiser, :shape [B, N, C, H, W]
    :params filters: filters used by model, :shape [B, L, N, K2, 1, Hf, Wf] with Hf = (H or 1) for L = number of cascaded filters
    """

    # -- file path --
    path = Path(f"./output/n2nwl/io_examples/{cfg.exp_name}/")
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
        arrows = create_arrow_image(directions[b],pad=2)
        tv_utils.save_image([arrows],fn)


    print(f"Wrote example images to file at [{path}]")
    plt.close("all")



def create_arrow_image(directions,pad=2):
    D = len(directions)
    assert D == 1,"Only one direction right now."
    W = 100
    S = (W + pad) * D + pad
    arrows = np.zeros((S,W+2*pad,3))
    direction = directions[0]
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


