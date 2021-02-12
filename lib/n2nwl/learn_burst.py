
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
from datasets.transform import ScaleZeroMean
from layers.ot_pytorch import sink_stabilized,sink,pairwise_distances
from layers.burst import BurstRecLoss

# -- [local] project code --
from .dist_loss import ot_pairwise_bp,ot_gaussian_bp,ot_pairwise2gaussian_bp

def train_loop(cfg,model,optimizer,criterion,train_loader,epoch,record_losses):


    # -=-=-=-=-=-=-=-=-=-=-
    #
    #    Setup for epoch
    #
    # -=-=-=-=-=-=-=-=-=-=-

    model.train()
    model = model.to(cfg.device)
    N = cfg.N
    total_loss = 0
    running_loss = 0
    szm = ScaleZeroMean()
    blocksize = 128
    unfold = torch.nn.Unfold(blocksize,1,0,blocksize)
    D = 5 * 10**3
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

    write_examples = True
    write_examples_iter = 800
    noise_level = cfg.noise_params['g']['stddev']

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Init Loss Functions
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    alignmentLossMSE = BurstRecLoss()
    denoiseLossMSE = BurstRecLoss()
    # denoiseLossOT = BurstResidualLoss()

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
    steps_per_epoch = D #len(train_loader)

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
        optimizer.zero_grad()
        model.zero_grad()
        model.denoiser_info.optim.zero_grad()

        # -- grab data batch --
        burst, res_imgs, raw_img, directions = next(train_iter)

        # -- getting shapes of data --
        N,BS,C,H,W = burst.shape
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

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #           Foward Pass
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        aligned,aligned_ave,denoised,denoised_ave,filters = model(burst)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #    Alignment Losses (MSE)
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        losses = alignmentLossMSE(aligned,aligned_ave,gt_img,cfg.global_step)
        ave_loss,burst_loss = [loss.item() for loss in losses]
        align_mse = np.sum(losses)
        align_mse_coeff = 0 #.933**cfg.global_step if cfg.global_step < 100 else 0

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #   Reconstruction Losses (MSE)
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        denoised_ave_d = denoised_ave.detach()
        losses = criterion(denoised,denoised_ave,gt_img,cfg.global_step)
        ave_loss,burst_loss = [loss.item() for loss in losses]
        rec_mse = np.sum(losses)
        rec_mse_coeff = 1.

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #    Reconstruction Losses (Distribution)
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- regularization scheduler --
        if cfg.global_step < 100: reg = 0.5
        elif cfg.global_step < 200: reg = 0.25
        elif cfg.global_step < 5000: reg = 0.15
        elif cfg.global_step < 10000: reg = 0.1
        else: reg = 0.05

        # -- computation --
        residuals = denoised - mid_img.unsqueeze(1).repeat(1,N,1,1,1)
        residuals = rearrange(residuals,'b n c h w -> b n (h w) c')
        rec_ot_pair_loss_v1 = ot_pairwise2gaussian_bp(residuals,K=6,reg=reg)
        # rec_ot_pair_loss_v2 = ot_pairwise_bp(residuals,K=3)
        rec_ot_pair_loss_v2 = torch.FloatTensor([0.]).to(cfg.device)
        rec_ot_pair = (rec_ot_pair_loss_v1 + rec_ot_pair_loss_v2)/2.
        rec_ot_pair_coeff = 100# - .997**cfg.global_step

            
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #              Final Losses
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        align_loss = align_mse_coeff * align_mse
        rec_loss = rec_ot_pair_coeff * rec_ot_pair + rec_mse_coeff * rec_mse
        final_loss = align_loss + rec_loss

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #              Record Keeping
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- alignment MSE --
        align_mse_losses += align_mse.item()
        align_mse_count += 1

        # -- reconstruction MSE --
        rec_mse_losses += rec_mse.item()
        rec_mse_count += 1

        # -- reconstruction Dist. --
        rec_ot_losses += rec_ot_pair.item()
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
        model.denoiser_info.optim.step()
        optimizer.step()


        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #            Printing to Stdout
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:

            # -- compute mse for fun --
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

            # -- update losses --
            running_loss /= cfg.log_interval

            # -- alignment MSE --
            align_mse_ave = align_mse_losses / align_mse_count
            align_mse_losses,align_mse_count = 0,0

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
                          psnr_aligned_std,psnr_misaligned_ave,psnr_misaligned_std,bm3d_nb_ave,bm3d_nb_std,
                          rec_mse_ave,rec_ot_ave)
            print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.2f +/- %2.2f [den]: %2.2f +/- %2.2f [al]: %2.2f +/- %2.2f [mis]: %2.2f +/- %2.2f [bm3d]: %2.2f +/- %2.2f [r-mse]: %.2e [r-ot]: %.2e" % write_info)
            running_loss = 0

        # -- write examples --
        if write_examples and (batch_idx % write_examples_iter) == 0 and (batch_idx > 0 or cfg.global_step == 0):
            write_input_output(cfg,stacked_burst,aligned,denoised,filters,directions)

        if use_timer: clock.toc()
        if use_timer: print(clock)
        cfg.global_step += 1
    total_loss /= len(train_loader)
    return total_loss,record_losses

def test_loop(cfg,model,criterion,test_loader,epoch):
    model.eval()
    model = model.to(cfg.device)
    total_psnr = 0
    total_loss = 0
    psnrs = np.zeros( (len(test_loader),cfg.batch_size) )
    use_record = False
    record_test = pd.DataFrame({'psnr':[]})

    with torch.no_grad():
        for batch_idx, (burst, res_imgs, raw_img, directions) in enumerate(test_loader):
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
            burst = burst.cuda(non_blocking=True)
            stacked_burst = torch.stack([burst[input_order[x]] for x in range(cfg.input_N)],dim=1)
            cat_burst = torch.cat([burst[input_order[x]] for x in range(cfg.input_N)],dim=1)
    
            # -- denoising --
            aligned,aligned_ave,denoised,denoised_ave,filters = model(burst)
            denoised_ave = denoised_ave.detach()

            # if not cfg.input_with_middle_frame:
            #     denoised_ave = model(cat_burst,stacked_burst)[1]
            # else:
            #     denoised_ave = model(cat_burst,stacked_burst)[0][middle_img_idx]

            # denoised_ave = burst[middle_img_idx] - rec_res
            
            # -- compare with stacked targets --
            denoised_ave = rescale_noisy_image(denoised_ave)        

            # -- compute psnr --
            loss = F.mse_loss(raw_img,denoised_ave,reduction='none').reshape(BS,-1)
            # loss = F.mse_loss(raw_img,burst[cfg.input_N//2]+0.5,reduction='none').reshape(BS,-1)
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

    # -- save histogram of residuals --
    plot_historgram_batch(burst,cfg.global_step,path,rand_name=True)

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


