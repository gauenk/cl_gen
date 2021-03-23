
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

# -- [local] project code --
from .dist_loss import ot_pairwise_bp
from .utils import EMA

def train_loop(cfg,model_target,model_online,optim_target,optim_online,criterion,train_loader,epoch,record_losses):


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #     setup for train epoch
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


    # -- setup for training --
    model_online.train()
    model_online = model_online.to(cfg.device)

    model_target.train()
    model_target = model_target.to(cfg.device)

    moving_average_decay = 0.99
    ema_updater = EMA(moving_average_decay)
    
    # -- init vars --
    N = cfg.N
    total_loss = 0
    running_loss = 0
    szm = ScaleZeroMean()
    blocksize = 128
    unfold = torch.nn.Unfold(blocksize,1,0,blocksize)
    D = 5 * 10**3
    use_record = False
    if record_losses is None: record_losses = pd.DataFrame({'burst':[],'ave':[],'ot':[],'psnr':[],'psnr_std':[]})
    nc_losses,nc_count = 0,0
    al_ot_losses,al_ot_count = 0,0
    rec_ot_losses,rec_ot_count = 0,0
    write_examples = True
    write_examples_iter = 800
    noise_level = cfg.noise_params['g']['stddev']
    one = torch.FloatTensor([1.]).to(cfg.device)
    switch = True

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #       run training epoch
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    for batch_idx, (burst, res_imgs, raw_img, directions) in enumerate(train_loader):
        if batch_idx > D: break

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #        forward pass
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- zero gradient --
        optim_online.zero_grad()
        optim_target.zero_grad()
        model_online.zero_grad()
        model_online.denoiser_info.optim.zero_grad()
        model_target.zero_grad()
        model_target.denoiser_info.optim.zero_grad()

        # -- reshaping of data --
        N,BS,C,H,W = burst.shape
        burst = burst.cuda(non_blocking=True)
        stacked_burst = rearrange(burst,'n b c h w -> b n c h w')

        # -- create target image --
        mid_img = burst[N//2]
        raw_zm_img = szm(raw_img.cuda(non_blocking=True))
        if cfg.supervised: t_img = szm(raw_img.cuda(non_blocking=True))
        else: t_img = burst[N//2]
        
        # -- direct denoising --
        aligned_o,aligned_ave_o,denoised_o,rec_img_o,filters_o = model_online(burst)
        aligned_t,aligned_ave_t,denoised_t,rec_img_t,filters_t = model_target(burst)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #        alignment losses
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


        # -- compute aligned losses to optimize --
        rec_img_d_o = rec_img_o.detach()
        losses = criterion(aligned_o,aligned_ave_o,rec_img_d_o,t_img,raw_zm_img,cfg.global_step)
        nc_loss,ave_loss,burst_loss,ot_loss = [loss.item() for loss in losses]
        kpn_loss = losses[1] + losses[2] # np.sum(losses)
        kpn_coeff = .9997**cfg.global_step

        # -- OT loss --
        al_ot_loss = torch.FloatTensor([0.]).to(cfg.device)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #    reconstruction losses
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            
        # -- decaying rec loss --
        rec_mse_coeff = 0.997**cfg.global_step
        rec_mse_loss = F.mse_loss(rec_img_o,mid_img)

        # -- BYOL loss --
        byol_loss = F.mse_loss(rec_img_o,rec_img_t)

        # -- OT loss -- 
        rec_ot_coeff = 100
        residuals = denoised_o - mid_img.unsqueeze(1).repeat(1,N,1,1,1)
        residuals = rearrange(residuals,'b n c h w -> b n (h w) c')
        # rec_ot_loss = ot_pairwise_bp(residuals,K=3)
        rec_ot_loss = torch.FloatTensor([0.]).to(cfg.device)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #    final losses & recording
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- final losses --
        align_loss = kpn_coeff * kpn_loss
        denoise_loss = rec_ot_coeff * rec_ot_loss + byol_loss  + rec_mse_coeff * rec_mse_loss

        # -- update alignment kl loss info --
        al_ot_losses += al_ot_loss.item()
        al_ot_count += 1

        # -- update reconstruction kl loss info --
        rec_ot_losses += rec_ot_loss.item()
        rec_ot_count += 1

        # -- update info --
        if not np.isclose(nc_loss,0):
            nc_losses += nc_loss
            nc_count += 1
        running_loss += align_loss.item() + denoise_loss.item()
        total_loss += align_loss.item() + denoise_loss.item()

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #    backprop and optimize
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- compute the gradients! --
        loss = align_loss + denoise_loss
        loss.backward()

        # -- backprop for [online] --
        optim_online.step()
        model_online.denoiser_info.optim.step()

        # -- exponential moving average for [target] --
        update_moving_average(ema_updater,model_target,model_online)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #      message to stdout
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:
            # -- compute mse for fun --
            BS = raw_img.shape[0]            
            raw_img = raw_img.cuda(non_blocking=True)

            # -- psnr for [average of aligned frames] --
            mse_loss = F.mse_loss(raw_img,aligned_ave_o+0.5,reduction='none').reshape(BS,-1)
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
                bm3d_rec = bm3d.bm3d(mid_img[b].cpu().transpose(0,2)+0.5, sigma_psd=noise_level/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                bm3d_rec = torch.FloatTensor(bm3d_rec).transpose(0,2)
                b_loss = F.mse_loss(raw_img[b].cpu(),bm3d_rec,reduction='none').reshape(BS,-1)
                b_loss = torch.mean(b_loss,1).detach().cpu().numpy()
                bm3d_nb_psnr = np.mean(mse_to_psnr(b_loss))
                bm3d_nb_psnrs.append(bm3d_nb_psnr)
            bm3d_nb_ave = np.mean(bm3d_nb_psnrs)
            bm3d_nb_std = np.std(bm3d_nb_psnrs)

            # -- psnr for aligned + denoised --
            raw_img_repN = raw_img.unsqueeze(1).repeat(1,N,1,1,1)
            mse_loss = F.mse_loss(raw_img_repN,denoised_o+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_denoised_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_denoised_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [model output image] --
            mse_loss = F.mse_loss(raw_img,rec_img_o+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr = np.mean(mse_to_psnr(mse_loss))
            psnr_std = np.std(mse_to_psnr(mse_loss))

            # -- write record --
            if use_record:
                record_losses = record_losses.append({'burst':burst_loss,'ave':ave_loss,'ot':ot_loss,'psnr':psnr,'psnr_std':psnr_std},ignore_index=True)
                
            # -- update losses --
            running_loss /= cfg.log_interval
            ave_nc_loss = nc_losses / nc_count if nc_count > 0 else 0

            # -- alignment kl loss --
            ave_al_ot_loss = al_ot_losses / al_ot_count if al_ot_count > 0 else 0
            al_ot_losses,al_ot_count = 0,0

            # -- reconstruction kl loss --
            ave_rec_ot_loss = rec_ot_losses / rec_ot_count if rec_ot_count > 0 else 0
            rec_ot_losses,rec_ot_count = 0,0


            # -- write to stdout --
            write_info = (epoch, cfg.epochs, batch_idx,len(train_loader),running_loss,psnr,psnr_std,
                          psnr_denoised_ave,psnr_denoised_std,psnr_aligned_ave,psnr_aligned_std,
                          psnr_misaligned_ave,psnr_misaligned_std,bm3d_nb_ave,bm3d_nb_std,
                          ave_nc_loss,ave_rec_ot_loss,ave_al_ot_loss)
            print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.2f +/- %2.2f [denoised]: %2.2f +/- %2.2f [aligned]: %2.2f +/- %2.2f [misaligned]: %2.2f +/- %2.2f [bm3d]: %2.2f +/- %2.2f [loss-nc]: %.2e [loss-rot]: %.2e [loss-aot]: %.2e" % write_info)
            running_loss = 0

        # -- write examples --
        if write_examples and (batch_idx % write_examples_iter) == 0 and (batch_idx > 0 or cfg.global_step == 0):
            write_input_output(cfg,stacked_burst,aligned_o,denoised_o,filters_o,directions)

        cfg.global_step += 1
    total_loss /= len(train_loader)
    return total_loss,record_losses

def update_moving_average(ema_updater, model_target, model_online):
    """
    ema_updater: Exponential moving average updater. 
    model_target: The model with ema parameters. 
    model_online: The model online uses sgd.
    """
    for params_online, params_target in zip(model_online.parameters(), model_target.parameters()):
        online_weights, target_weights = params_online.data, params_target.data
        params_target.data = ema_updater.update_average(online_weights, target_weights)

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
            aligned,aligned_ave,denoised,rec_img,filters = model(burst)
            rec_img = rec_img.detach()

            # if not cfg.input_with_middle_frame:
            #     rec_img = model(cat_burst,stacked_burst)[1]
            # else:
            #     rec_img = model(cat_burst,stacked_burst)[0][middle_img_idx]

            # rec_img = burst[middle_img_idx] - rec_res
            
            # -- compare with stacked targets --
            rec_img = rescale_noisy_image(rec_img)        

            # -- compute psnr --
            loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
            # loss = F.mse_loss(raw_img,burst[cfg.input_N//2]+0.5,reduction='none').reshape(BS,-1)
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
            if batch_idx % 250 == 0: print("[%d/%d] Test PSNR: %2.2f" % (batch_idx,len(test_loader),total_psnr / (batch_idx+1)))

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
    path = Path(f"./output/byol/io_examples/{cfg.exp_name}/")
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


