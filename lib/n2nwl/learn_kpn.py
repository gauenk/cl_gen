
# -- python imports --
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
from datasets.transforms import ScaleZeroMean
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
    if record_losses is None: record_losses = pd.DataFrame({'kpn':[],'ot':[],'psnr':[],'psnr_std':[]})

    # if cfg.N != 5: return
    switch = True
    for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(train_loader):
        if batch_idx > D: break

        optimizer.zero_grad()
        model.zero_grad()

        # fig,ax = plt.subplots(figsize=(10,10))
        # imgs = burst_imgs + 0.5
        # imgs.clamp_(0.,1.)
        # raw_img = raw_img.expand(burst_imgs.shape)
        # print(imgs.shape,raw_img.shape)
        # all_img = torch.cat([imgs,raw_img],dim=1)
        # print(all_img.shape)
        # grids = [vutils.make_grid(all_img[i],nrow=16) for i in range(cfg.dynamic.frames)]
        # ims = [[ax.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in grids]
        # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save(f"{settings.ROOT_PATH}/train_loop_voc.mp4", writer=writer)
        # print("I DID IT!")
        # return

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
        if cfg.blind:
            t_img = burst_imgs[middle_img_idx]
        else:
            t_img = szm(raw_img.cuda(non_blocking=True))
        
        # -- direct denoising --
        rec_img_i,rec_img = model(cat_burst,stacked_burst)

        # rec_img = burst_imgs[middle_img_idx] - rec_res

        # -- compare with stacked burst --
        # print(cfg.blind,t_img.min(),t_img.max(),t_img.mean())
        # rec_img = rec_img.expand(t_img.shape)
        # loss = F.mse_loss(t_img,rec_img)


        # -- compute mse to optimize --
        mse_loss = F.mse_loss(rec_img,t_img)

        # -- compute kpn loss to optimize --
        kpn_losses = criterion(rec_img_i, rec_img, t_img, cfg.global_step)
        kpn_loss = np.sum(kpn_losses)

        # -- compute blockwise differences --
        rec_img_i_bn = rearrange(rec_img_i,'b n c h w -> (b n) c h w')
        r_middle_img = t_img.unsqueeze(1).repeat(1,N,1,1,1)
        r_middle_img = rearrange(r_middle_img,'b n c h w -> (b n) c h w')
        diffs = r_middle_img - rec_img_i_bn
        # diffs = rearrange(unfold(diffs),'(b n) (c i) r -> b n r (c i)',b=BS,c=3)

        # -- compute OT loss --
        # mse_loss = torch.mean(torch.pow(diffs,2))
        diffs = rearrange(diffs,'(b n) c h w -> b n (h w) c',n=N)
        ot_loss = 0
        #skip_middle = i != N//2 and j != N//2
        pairs = list(set([(i,j) for i in range(N) for j in range(N) if i<j]))
        P = len(pairs)
        S = 3 #P
        r_idx = npr.choice(range(P),S)
        for idx in r_idx:
            i,j = pairs[idx]
            if i >= j: continue
            # assert BS == 1, "batch size must be one right now."
            for b in range(BS):
                di,dj = diffs[b,i],diffs[b,j]
                M = torch.sum(torch.pow(di.unsqueeze(1) - dj,2),dim=-1)
                ot_loss += sink_stabilized(M, 0.5)
        ot_loss /= S*BS

        # M = torch.mean(torch.pow(diffs.unsqueeze(1) - diffs,2),dim=2)
        # ot_loss = sink(M, 0.5)

        
        # -- compute stats for each block --
        # mean_est = torch.mean(diffs, dim=(1,2,3), keepdim=True)
        # std_est = torch.pow( diffs - mean_est, 2)
        # # mse_loss = F.mse_loss(r_middle_img,rec_img_i_bn,reduction='none')
        # std_est = torch.flatten(torch.mean( std_est, dim=(1,2,3) ))
        # # dist_loss = torch.norm(std_est.unsqueeze(1) - std_est)

        # # -- flatten and compare each block stats --
        # dist_loss = 0
        # mean_est = torch.flatten(mean_est)
        # std_est = torch.flatten(std_est)
        # M = mean_est.shape[0]
        # for i in range(M):
        #     for j in range(M):
        #         if i >= j: continue
        #         si,sj = std_est[i],std_est[j]
        #         dist_loss += torch.abs(mean_est[i] - mean_est[j])
        #         dist_loss += torch.abs(si + sj - 2 * (si * sj)**0.5)

        # -- combine loss --
        # print(kpn_loss.item(),10**3 * ot_loss.item(),ot_loss.item() / (1 + mse_loss.item()))
        # loss = kpn_loss + 10**4 * ot_loss / (1 + mse_loss.item())
        alpha,beta = criterion.loss_anneal.alpha,criterion.loss_anneal.beta
        ot_coeff = 10
        # loss = kpn_loss
        loss = kpn_loss + ot_coeff * ot_loss# / (1 + mse_loss.item())
        # print(kpn_loss.item(), 10**4 * ot_loss.item() / (1 + mse_loss.item()))

        # loss = mse_loss + ot_loss / (1 +  mse_loss.item())
        # if batch_idx % 100 == 0 or switch: switch = not switch
        # if switch:
        #     loss += kpn_loss# + ot_loss / (1 + kpn_loss.item())
        #     # loss = kpn_loss + ot_loss / (1 + kpn_loss.item())
        # print(ot_loss.item(),mse_loss.item(),kpn_loss.item(),loss.item())
            
        # -- update info --
        running_loss += loss.item()
        total_loss += loss.item()

        # -- BP and optimize --
        loss.backward()
        optimizer.step()

        if True:
            # -- compute mse for fun --
            BS = raw_img.shape[0]            
            raw_img = raw_img.cuda(non_blocking=True)
            mse_loss = F.mse_loss(raw_img,rec_img+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr = np.mean(mse_to_psnr(mse_loss))
            psnr_std = np.std(mse_to_psnr(mse_loss))
            record_losses = record_losses.append({'kpn':kpn_loss.item(),'ot':ot_loss.item(),'psnr':psnr,'psnr_std':psnr_std},ignore_index=True)
            running_loss /= cfg.log_interval
            if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:
                print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.2f +/- %2.2f"%(epoch, cfg.epochs, batch_idx,
                                                                       len(train_loader),
                                                                       running_loss,psnr,psnr_std))
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
            rec_img = model(cat_burst,stacked_burst)[1].detach()

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
