
# -- python imports --
import bm3d
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
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transform import ScaleZeroMean,RandomChoice
from layers.ot_pytorch import sink_stabilized,sink,pairwise_distances
from layers.burst import BurstRecLoss,EntropyLoss

# -- [local] project code --
from n2nwl.dist_loss import ot_pairwise_bp,ot_gaussian_bp,ot_pairwise2gaussian_bp,kl_gaussian_bp,w_gaussian_bp,kl_gaussian_bp_patches
from n2nwl.misc import AlignmentFilterHooks
from n2nwl.plot import plot_histogram_residuals_batch,plot_histogram_gradients,plot_histogram_gradient_norms
from .sim_search import compute_similar_bursts,compute_similar_bursts_n2sim,create_k_grid,compare_sim_patches_methods,compare_sim_images_methods

def train_loop(cfg,model,scheduler,train_loader,epoch,record_losses):


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
    rec_mse_losses,rec_mse_count = 0,0
    rec_ot_losses,rec_ot_count = 0,0
    running_loss,total_loss = 0,0

    write_examples = False
    write_examples_iter = 800
    noise_level = cfg.noise_params['g']['stddev']

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
    #      Init Loss Functions
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    alignmentLossMSE = BurstRecLoss()
    denoiseLossMSE = BurstRecLoss(alpha=1.0,gradient_L1=~cfg.supervised)
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
    #    Final Configs
    #
    # -=-=-=-=-=-=-=-=-=-=-

    use_timer = False
    one = torch.FloatTensor([1.]).to(cfg.device)
    switch = True
    if use_timer: clock = Timer()
    K = 8
    train_iter = iter(train_loader)
    steps_per_epoch = len(train_loader)
    write_examples_iter = steps_per_epoch//3
    all_filters = []

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

        # -- grab data batch --
        burst, res_imgs, raw_img, directions = next(train_iter)

        # -- getting shapes of data --
        N,B,C,H,W = burst.shape
        burst = burst.cuda(non_blocking=True)
        raw_zm_img = szm(raw_img.cuda(non_blocking=True))
        burst_og = burst.clone()
        mid_img_og = burst[N//2]

        # -- spoof noise2noise --
        # assert K == 1, "spoofing noise2noise requires K == 1 patches"
        # sim_burst = raw_zm_img.unsqueeze(0).unsqueeze(2).repeat(1,1,K,1,1,1)
        # sim_burst = torch.normal(sim_burst,noise_level/255.)
        # sim_burst = torch.cat([burst.unsqueeze(2),sim_burst],dim=2)
        # k_ins = np.zeros(K).astype(np.int)
        # k_outs = np.arange(K)+1

        # -- get similar images --
        def compare_sim_computation_times():
            # compare_sim_patches_methods(cfg,burst,K,patchsize=3)
            # compare_sim_images_methods(cfg,burst,K,patchsize=3)
            t_a,t_b = Timer(),Timer()
            t_a.tic()
            sim_bursts_a = compute_similar_bursts_n2sim(cfg,burst,K,patchsize=3)
            t_a.toc()

            t_b.tic()
            sim_burst = compute_similar_bursts(cfg,burst,K,patchsize=3)
            t_b.toc()

            print(t_a)
            print(t_b)

        # compare_sim_computation_times()
        sim_burst = compute_similar_bursts(cfg,burst,K,patchsize=3)
        k_ins,k_outs = create_k_grid(sim_burst,K+1,shuffle=True,L=K)

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
            burst = sim_burst[:,:,k_in]
            mid_img =  sim_burst[N//2,:,k_out]
            # mid_img =  sim_burst[N//2,:]
            # print(burst.shape,mid_img.shape)
            # print(F.mse_loss(burst,mid_img).item())
            if cfg.supervised: gt_img = raw_zm_img
            else: gt_img = mid_img
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
            aligned,aligned_ave,denoised,denoised_ave = outputs[:4]
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
                filters_shaped = rearrange(filters,'b n k2 c h w -> (b n c h w) k2',n=N)
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
    
            losses = denoiseLossMSE(denoised,denoised_ave,gt_img,cfg.global_step)
            losses = [ one, one ]
            # ave_loss,burst_loss = [loss.item() for loss in losses]
            rec_mse = np.sum(losses)
            rec_mse = F.mse_loss(denoised_ave,gt_img)
            rec_mse_coeff = 1.
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #
            #    Reconstruction Losses (Distribution)
            #
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            residuals = denoised - gt_img.unsqueeze(1).repeat(1,N,1,1,1)
            rec_ot = torch.FloatTensor([0.]).to(cfg.device)
            # rec_ot = kl_gaussian_bp_patches(residuals,noise_level,flip=True,patchsize=16)
            if torch.any(torch.isnan(rec_ot)): rec_ot = torch.FloatTensor([0.]).to(cfg.device)
            if torch.any(torch.isinf(rec_ot)): rec_ot = torch.FloatTensor([0.]).to(cfg.device)
            # print(rec_ot)
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
            final_loss.backward()
    
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
            aligned,aligned_ave,denoised,denoised_ave = outputs[:4]
            aligned_filters,denoised_filters = outputs[4:]

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
                bm3d_rec = bm3d.bm3d(mid_img_og[b].cpu().transpose(0,2)+0.5,
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


