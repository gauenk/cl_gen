
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

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as tv_utils

# -- project code --
import settings
from pyutils.timer import Timer
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transforms import ScaleZeroMean
from layers.ot_pytorch import sink_stabilized,sink,pairwise_distances
from layers.burst import LossRec,LossRecBurst

# -- [local] project code --
from .dist_loss import ot_pairwise_bp,ot_gaussian_bp,ot_pairwise2gaussian_bp

def test_loop(cfg,model,criterion,test_loader,epoch):
    return test_loop_offset(cfg,model,criterion,test_loader,epoch)

def train_loop(cfg,model,noise_critic,optimizer,criterion,train_loader,epoch,record_losses):

    # -=-=-=-=-=-=-=-=-=-=-
    #    Setup for epoch
    # -=-=-=-=-=-=-=-=-=-=-

    model.train()
    model = model.to(cfg.device)
    N = cfg.N
    szm = ScaleZeroMean()
    blocksize = 128
    unfold = torch.nn.Unfold(blocksize,1,0,blocksize)
    D = 5 * 10**3
    use_record = False
    if record_losses is None: record_losses = pd.DataFrame({'burst':[],'ave':[],'ot':[],'psnr':[],'psnr_std':[]})
    write_examples = True
    write_examples_iter = 800
    noise_level = cfg.noise_params['g']['stddev']

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #      Init Record Keeping
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    losses_nc,losses_nc_count = 0,0
    losses_mse,losses_mse_count = 0,0
    running_loss,total_loss = 0,0

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #      Init Loss Functions
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    lossRecMSE = LossRec(tensor_grad=cfg.supervised)
    lossBurstMSE = LossRecBurst(tensor_grad=cfg.supervised)

    # -=-=-=-=-=-=-=-=-=-=-
    #    Final Configs
    # -=-=-=-=-=-=-=-=-=-=-

    use_timer = False
    one = torch.FloatTensor([1.]).to(cfg.device)
    switch = True
    train_iter = iter(train_loader)
    if use_timer: clock = Timer()

    # -=-=-=-=-=-=-=-=-=-=-
    #    GAN Scheduler
    # -=-=-=-=-=-=-=-=-=-=-

    # -- noise critic steps --
    if epoch == 0: disc_steps = 0
    elif epoch < 3: disc_steps = 1
    elif epoch < 10: disc_steps = 1
    else: disc_steps = 1

    # -- denoising steps --
    if epoch == 0: gen_steps = 1
    if epoch < 3: gen_steps = 15
    if epoch < 10: gen_steps = 10
    else: gen_steps = 10

    # -- steps each epoch --
    steps_per_iter = disc_steps * gen_steps
    steps_per_epoch = len(train_loader) // steps_per_iter
    if steps_per_epoch > 120: steps_per_epoch = 120

    # -=-=-=-=-=-=-=-=-=-=-
    #    Start Epoch
    # -=-=-=-=-=-=-=-=-=-=-

    for batch_idx in range(steps_per_epoch):


        for gen_step in range(gen_steps):

            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #      Setting up for Iteration
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            # -- setup iteration timer --
            if use_timer: clock.tic()
    
            # -- zero gradients --
            optimizer.zero_grad()
            model.zero_grad()
            model.denoiser_info.model.zero_grad()
            model.denoiser_info.optim.zero_grad()
            noise_critic.disc.zero_grad()
            noise_critic.optim.zero_grad()
    
            # -- grab data batch --
            burst, res_imgs, raw_img, directions = next(train_iter)
    
            # -- getting shapes of data --
            N,BS,C,H,W = burst.shape
            burst = burst.cuda(non_blocking=True)
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #      Formatting Images for FP
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            # -- creating some transforms --
            stacked_burst = rearrange(burst,'n b c h w -> b n c h w')
            cat_burst = rearrange(burst,'n b c h w -> (b n) c h w')
    
            # -- extract target image --
            mid_img =  burst[N//2]
            raw_zm_img = szm(raw_img.cuda(non_blocking=True))
            if cfg.supervised: gt_img = raw_zm_img
            else: gt_img = mid_img
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #           Foward Pass
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            aligned,aligned_ave,denoised,denoised_ave,filters = model(burst)
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #    MSE (KPN) Reconstruction Loss
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            loss_rec = lossRecMSE(denoised_ave,gt_img)
            loss_burst = lossBurstMSE(denoised,gt_img)
            loss_mse = loss_rec + 100 * loss_burst
            gbs,spe = cfg.global_step,steps_per_epoch
            if epoch < 3: weight_mse = 10
            else: weight_mse = 10 * 0.9999**(gbs - 3*spe)
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #      Noise Critic Loss
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            loss_nc = noise_critic.compute_residual_loss(denoised,gt_img)
            weight_nc = 1
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #          Final Loss
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            final_loss = weight_mse * loss_mse + weight_nc * loss_nc
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #    Update Info for Record Keeping
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            # -- update alignment kl loss info --
            losses_nc += loss_nc.item()
            losses_nc_count += 1
    
            # -- update reconstruction kl loss info --
            losses_mse += loss_mse.item()
            losses_mse_count += 1
    
            # -- update info --
            running_loss += final_loss.item()
            total_loss += final_loss.item()
    
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            #          Backward Pass
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
            # -- compute the gradients! --
            final_loss.backward()
    
            # -- backprop now. --
            model.denoiser_info.optim.step()
            optimizer.step()

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #     Iterate for Noise Critic
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        for disc_step in range(disc_steps):

            # -- zero gradients --
            optimizer.zero_grad()
            model.zero_grad()
            model.denoiser_info.optim.zero_grad()
            noise_critic.disc.zero_grad()
            noise_critic.optim.zero_grad()
    
            # -- grab noisy data --
            _burst, _res_imgs, _raw_img, _directions = next(train_iter)
            _burst = _burst.to(cfg.device)
            
            # -- generate "fake" data from noisy data --
            _aligned,_aligned_ave,_denoised,_denoised_ave,_filters = model(_burst)
            _residuals = _denoised - _burst[N//2].unsqueeze(1).repeat(1,N,1,1,1)

            # -- update discriminator --
            loss_disc = noise_critic.update_disc(_residuals)

            # -- message to stdout --
            first_update = (disc_step == 0)
            last_update = (disc_step == disc_steps-1)
            iter_update = first_update or last_update
            # if (batch_idx % cfg.log_interval//2) == 0 and batch_idx > 0 and iter_update:
            print(f"[Noise Critic]: {loss_disc}")

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #      Print Message to Stdout
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:

            # -- init --
            BS = raw_img.shape[0]            
            raw_img = raw_img.cuda(non_blocking=True)

            # -- psnr for [average of aligned frames] --
            mse_loss = F.mse_loss(raw_img,aligned_ave+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_aligned_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_aligned_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [average of input, misaligned frames] --
            mis_ave = torch.mean(stacked_burst,dim=1)
            mse_loss = F.mse_loss(raw_img,mis_ave+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_misaligned_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_misaligned_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [bm3d] --
            bm3d_nb_psnrs = []
            for b in range(BS):
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
            mse_loss = F.mse_loss(raw_img_repN,denoised+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_denoised_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_denoised_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [model output image] --
            mse_loss = F.mse_loss(raw_img,denoised_ave+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr = np.mean(mse_to_psnr(mse_loss))
            psnr_std = np.std(mse_to_psnr(mse_loss))

            # -- write record --
            if use_record:
                record_losses = record_losses.append({'burst':burst_loss,'ave':ave_loss,'ot':ot_loss,'psnr':psnr,'psnr_std':psnr_std},ignore_index=True)
                
            # -- update losses --
            running_loss /= cfg.log_interval

            # -- average mse losses --
            ave_losses_mse = losses_mse / losses_mse_count
            losses_mse,losses_mse_count = 0,0

            # -- average noise critic loss --
            ave_losses_nc = losses_nc / losses_nc_count
            losses_nc,losses_nc_count = 0,0

            # -- write to stdout --
            write_info = (epoch, cfg.epochs, batch_idx, steps_per_epoch,
                          running_loss,psnr,psnr_std,psnr_denoised_ave,
                          psnr_denoised_std,psnr_misaligned_ave,psnr_misaligned_std,
                          bm3d_nb_ave,bm3d_nb_std,ave_losses_mse,ave_losses_nc)
            print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.2f +/- %2.2f [den]: %2.2f +/- %2.2f [mis]: %2.2f +/- %2.2f [bm3d]: %2.2f +/- %2.2f [mse]: %.2e [nc]: %.2e" % write_info)
            running_loss = 0

        # -- write examples --
        if write_examples and (batch_idx % write_examples_iter) == 0 and (batch_idx > 0 or cfg.global_step == 0):
            write_input_output(cfg,stacked_burst,aligned,denoised,filters,directions)

        if use_timer: clock.toc()
        if use_timer: print(clock)
        cfg.global_step += 1
    total_loss /= len(train_loader)
    return total_loss,record_losses

def test_loop_offset(cfg,model,criterion,test_loader,epoch):
    model.eval()
    model = model.to(cfg.device)
    total_psnr = 0
    total_loss = 0
    psnrs = np.zeros( (len(test_loader),cfg.batch_size) )
    use_record = False
    record_test = pd.DataFrame({'psnr':[]})

    with torch.no_grad():
        for batch_idx, (burst_imgs, res_imgs, raw_img, directions) in enumerate(test_loader):
            BS = raw_img.shape[0]
            
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
            burst_imgs = burst_imgs.cuda(non_blocking=True)
            stacked_burst = torch.stack([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
            cat_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
    
            # -- denoising --
            aligned,aligned_ave,denoised,rec_img,filters = model(burst_imgs)
            rec_img = rec_img.detach()

            # if not cfg.input_with_middle_frame:
            #     rec_img = model(cat_burst,stacked_burst)[1]
            # else:
            #     rec_img = model(cat_burst,stacked_burst)[0][middle_img_idx]

            # rec_img = burst_imgs[middle_img_idx] - rec_res
            
            # -- compare with stacked targets --
            rec_img = rescale_noisy_image(rec_img)        

            # -- compute psnr --
            loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
            # loss = F.mse_loss(raw_img,burst_imgs[cfg.input_N//2]+0.5,reduction='none').reshape(BS,-1)
            loss = torch.mean(loss,1).detach().cpu().numpy()
            psnr = mse_to_psnr(loss)
            psnrs[batch_idx,:] = psnr
                        
            if use_record:
                record_test = record_test.append({'psnr':psnr},ignore_index=True)
            total_psnr += np.mean(psnr)
            total_loss += np.mean(loss)

            # if (batch_idx % cfg.test_log_interval) == 0:
            #     root = Path(f"{settings.ROOT_PATH}/output/n2n/offset_out_noise/rec_imgs/N{cfg.N}/e{epoch}")
            #     if not root.exists(): root.mkdir(parents=True)
            #     fn = root / Path(f"b{batch_idx}.png")
            #     nrow = int(np.sqrt(cfg.batch_size))
            #     rec_img = rec_img.detach().cpu()
            #     grid_imgs = tv_utils.make_grid(rec_img, padding=2, normalize=True, nrow=nrow)
            #     plt.imshow(grid_imgs.permute(1,2,0))
            #     plt.savefig(fn)
            #     plt.close('all')
            if batch_idx % 100 == 0: print("[%d/%d] Test PSNR: %2.2f" % (batch_idx,len(test_loader),total_psnr / (batch_idx+1)))

    psnr_ave = np.mean(psnrs)
    psnr_std = np.std(psnrs)
    ave_loss = total_loss / len(test_loader)
    print("[N: %d] Testing: [psnr: %2.2f +/- %2.2f] [ave loss %2.3e]"%(cfg.N,psnr_ave,psnr_std,ave_loss))
    return psnr_ave,record_test


def write_input_output(cfg,burst,aligned,denoised,filters,directions):

    """
    :params burst: input images to the model, :shape [B, N, C, H, W]
    :params aligned: output images from the model, :shape [B, N, C, H, W]
    :params filters: filters used by model, :shape [B, N, K2, 1, Hf, Wf] with Hf = (H or 1)
    """

    # -- file path --
    path = Path(f"./output/n2nwl/io_examples/{cfg.exp_name}/")
    if not path.exists(): path.mkdir(parents=True)

    # -- init --
    B,N,C,H,W = burst.shape

    # -- save file per burst --
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
        K = int(np.sqrt(filters.shape[2]))
        if filters.shape[-1] > 1:
            S = npr.permutation(filters.shape[-1])[:10]
            filters_b = filters[b,:,:,0,S,S].view(N*10,1,K,K)
        else: filters_b = filters[b,:,:,0,0,0].view(N,1,K,K)
        tv_utils.save_image(filters_b,fn,nrow=N,normalize=True)

        # -- save direction image --
        fn = path / Path(f"arrows_{cfg.global_step}_{b}.png")
        arrows = create_arrow_image(directions[b],pad=2)
        tv_utils.save_image([arrows],fn)


    print(f"Wrote example images to file at [{path}]")



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


