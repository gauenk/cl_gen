
# -- python imports --
import bm3d
import pandas as pd
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange, repeat, reduce

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as vutils


# -- project code --
import settings
from pyutils.timer import Timer
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transform import ScaleZeroMean
from layers.ot_pytorch import sink_stabilized,sink,pairwise_distances


def train_loop(cfg,model,optimizer,criterion,train_loader,epoch,record_losses):
    return train_loop_offset(cfg,model,optimizer,criterion,train_loader,epoch,record_losses)

def test_loop(cfg,model,criterion,test_loader,epoch):
    return test_loop_offset(cfg,model,criterion,test_loader,epoch)


def train_loop_offset(cfg,model,optimizer,criterion,train_loader,epoch,record_losses):
    model.train()
    model = model.to(cfg.device)
    N = cfg.N
    total_loss = 0
    running_loss = 0
    szm = ScaleZeroMean()
    blocksize = 128
    unfold = torch.nn.Unfold(blocksize,1,0,blocksize)
    D = 5 * 10**3
    if record_losses is None: record_losses = pd.DataFrame({'burst':[],'ave':[],'ot':[],'psnr':[],'psnr_std':[]})
    nc_losses,nc_count = 0,0

    # if cfg.N != 5: return
    one = torch.FloatTensor([1.]).to(cfg.device)
    switch = True
    for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(train_loader):
        if batch_idx > D: break

        optimizer.zero_grad()
        model.zero_grad()

        # -- reshaping of data --
        # raw_img = raw_img.cuda(non_blocking=True)
        input_order = np.arange(cfg.N)
        # print("pre",input_order,cfg.blind,cfg.N)
        middle_img_idx = -1
        if not cfg.input_with_middle_frame:
            middle = len(input_order) // 2
            # print(middle)
            middle_img_idx = input_order[middle]
            # input_order = np.r_[input_order[:middle],input_order[middle+1:]]
        else:
            middle = len(input_order) // 2
            input_order = np.arange(cfg.N)
            middle_img_idx = input_order[middle]
            # input_order = np.arange(cfg.N)
        # print("post",input_order,cfg.blind,cfg.N,middle_img_idx)

        N,BS,C,H,W = burst_imgs.shape
        burst_imgs = burst_imgs.cuda(non_blocking=True)
        middle_img = burst_imgs[middle_img_idx]
        # print(cfg.N,cfg.blind,[input_order[x] for x in range(cfg.input_N)])
        # stacked_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
        # print("stacked_burst",stacked_burst.shape)
        # print("burst_imgs.shape",burst_imgs.shape)
        # print("stacked_burst.shape",stacked_burst.shape)

        # -- add input noise --
        burst_imgs_noisy = burst_imgs.clone()
        if cfg.input_noise:
            noise = np.random.rand() * cfg.input_noise_level
            if cfg.input_noise_middle_only:
                burst_imgs_noisy[middle_img_idx] = torch.normal(burst_imgs_noisy[middle_img_idx],noise)
            else:
                burst_imgs_noisy = torch.normal(burst_imgs_noisy,noise)

        # -- create inputs for kpn --
        stacked_burst = torch.stack([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
        cat_burst = torch.cat([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
        # print(stacked_burst.shape)
        # print(cat_burst.shape)

        # -- extract target image --
        raw_zm_img = szm(raw_img.cuda(non_blocking=True))
        if cfg.blind:
            t_img = burst_imgs[middle_img_idx]
        else:
            t_img = szm(raw_img.cuda(non_blocking=True))
        
        # -- direct denoising --
        aligned,aligned_ave,rec_img = model(burst_imgs)

        # -- compute burst loss to optimize --
        losses = criterion(aligned,aligned_ave,rec_img,t_img,raw_zm_img,cfg.global_step)
        nc_loss,ave_loss,burst_loss,ot_loss = [loss.item() for loss in losses]
        loss = np.sum(losses)
        # print(nc_loss,ave_loss,burst_loss,ot_loss)

            
        # -- update info --
        if not np.isclose(nc_loss,0):
            nc_losses += nc_loss
            nc_count += 1
        running_loss += loss.item()
        total_loss += loss.item()

        # -- BP and optimize --
        loss.backward()
        optimizer.step()

        # if True:
        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:
            # -- compute mse for fun --
            BS = raw_img.shape[0]            
            raw_img = raw_img.cuda(non_blocking=True)

            # -- psnr for [average of aligned frames] --
            mse_loss = F.mse_loss(raw_img,aligned_ave+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr_aligned_ave = np.mean(mse_to_psnr(mse_loss))
            psnr_aligned_std = np.std(mse_to_psnr(mse_loss))

            # -- psnr for [bm3d] --
            bm3d_nb_psnrs = []
            for b in range(BS):
                bm3d_rec = bm3d.bm3d(t_img[b].cpu().transpose(0,2)+0.5, sigma_psd=25/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                bm3d_rec = torch.FloatTensor(bm3d_rec).transpose(0,2)
                b_loss = F.mse_loss(raw_img[b].cpu(),bm3d_rec,reduction='none').reshape(BS,-1)
                b_loss = torch.mean(b_loss,1).detach().cpu().numpy()
                bm3d_nb_psnr = np.mean(mse_to_psnr(b_loss))
                bm3d_nb_psnrs.append(bm3d_nb_psnr)
            bm3d_nb_ave = np.mean(bm3d_nb_psnrs)
            bm3d_nb_std = np.std(bm3d_nb_psnrs)


            # -- psnr for [model output image] --
            mse_loss = F.mse_loss(raw_img,rec_img+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr = np.mean(mse_to_psnr(mse_loss))
            psnr_std = np.std(mse_to_psnr(mse_loss))

            # record_losses = record_losses.append({'burst':burst_loss,'ave':ave_loss,'ot':ot_loss,'psnr':psnr,'psnr_std':psnr_std},ignore_index=True)
            running_loss /= cfg.log_interval
            ave_nc_loss = nc_losses / nc_count if nc_count > 0 else 0
            if True:
            # if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:
                write_info = (epoch, cfg.epochs, batch_idx,len(train_loader),running_loss,psnr,psnr_std,
                              psnr_aligned_ave,psnr_aligned_std,bm3d_nb_ave,bm3d_nb_std,ave_nc_loss)
                print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.2f +/- %2.2f [Ave]: %2.2f +/- %2.2f [bm3d]: %2.2f +/- %2.2f [loss-nc]: %.2e" % write_info)
            running_loss = 0
        cfg.global_step += 1
    total_loss /= len(train_loader)
    return total_loss,record_losses

def test_loop_offset(cfg,model,criterion,test_loader,epoch):
    model.eval()
    model = model.to(cfg.device)
    total_psnr = 0
    total_loss = 0
    record_test = pd.DataFrame({'psnr':[]})

    with torch.no_grad():
        for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(test_loader):
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
            aligned,aligned_ave,rec_img = model(burst_imgs)
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
                        
            record_test = record_test.append({'psnr':psnr},ignore_index=True)
            total_psnr += np.mean(psnr)
            total_loss += np.mean(loss)

            # if (batch_idx % cfg.test_log_interval) == 0:
            #     root = Path(f"{settings.ROOT_PATH}/output/n2n/offset_out_noise/rec_imgs/N{cfg.N}/e{epoch}")
            #     if not root.exists(): root.mkdir(parents=True)
            #     fn = root / Path(f"b{batch_idx}.png")
            #     nrow = int(np.sqrt(cfg.batch_size))
            #     rec_img = rec_img.detach().cpu()
            #     grid_imgs = vutils.make_grid(rec_img, padding=2, normalize=True, nrow=nrow)
            #     plt.imshow(grid_imgs.permute(1,2,0))
            #     plt.savefig(fn)
            #     plt.close('all')
            if batch_idx % 1000 == 0: print("[%d/%d] Test PSNR: %2.2f" % (batch_idx,len(test_loader),total_psnr / (batch_idx+1)))

    ave_psnr = total_psnr / len(test_loader)
    ave_loss = total_loss / len(test_loader)
    print("[Blind: %d | N: %d] Testing results: Ave psnr %2.3e Ave loss %2.3e"%(cfg.blind,cfg.N,ave_psnr,ave_loss))
    return ave_psnr,record_test
