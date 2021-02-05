
# -- python imports --
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
import torchvision.transforms as th_trans

# -- project code --
import settings
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transform import ScaleZeroMean

def train_loop(cfg,model,optimizer,criterion,train_loader,epoch):
    return train_loop_offset(cfg,model,optimizer,criterion,train_loader,epoch)

def test_loop(cfg,model,criterion,test_loader,epoch):
    return test_loop_offset(cfg,model,criterion,test_loader,epoch)


def train_loop_offset(cfg,model,optimizer,criterion,train_loader,epoch):
    model.train()
    model = model.to(cfg.device)
    N = cfg.N
    total_loss = 0
    running_loss = 0
    szm = ScaleZeroMean()
    # random_eraser = th_trans.RandomErasing(scale=(0.40,0.80))
    random_eraser = th_trans.RandomErasing(scale=(0.02,0.33))

    # if cfg.N != 5: return
    # for batch_idx, (burst_imgs, raw_img) in enumerate(train_loader):
    for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(train_loader):


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

        # -- shape info --
        N,BS,C,H,W = burst_imgs.shape

        # -- reshaping of data --
        # raw_img = raw_img.cuda(non_blocking=True)
        input_order = np.arange(cfg.N)
        # print("pre",input_order,cfg.blind,cfg.N)
        middle_img_idx = -1
        if not cfg.input_with_middle_frame:
            middle = len(input_order) // 2
            # print(middle)
            middle_img_idx = input_order[middle]
            input_order = np.r_[input_order[:middle],input_order[middle+1:]]
        else:
            middle = len(input_order) // 2
            middle_img_idx = input_order[middle]
            input_order = np.arange(cfg.N)
        # print("post",input_order,middle_img_idx,cfg.blind,cfg.N)

        # -- add input noise --
        burst_imgs = burst_imgs.cuda(non_blocking=True) 
        middle_img = burst_imgs[middle_img_idx]
        burst_imgs_noisy = burst_imgs.clone()
        if cfg.input_noise:
            # noise = np.random.rand() * cfg.input_noise_level
            noise = cfg.input_noise_level
            if cfg.input_noise_middle_only:
                burst_imgs_noisy[middle_img_idx] = torch.normal(burst_imgs_noisy[middle_img_idx],noise)
            else:
                burst_imgs_noisy = torch.normal(burst_imgs_noisy,noise)

        # if cfg.middle_frame_random_erase:
        #     for i in range(burst_imgs_noisy[middle_img_idx].shape[0]):
        #         tmp = random_eraser(burst_imgs_noisy[middle_img_idx][i])
        #         burst_imgs_noisy[middle_img_idx][i] = tmp
        # burst_imgs_noisy = torch.normal(burst_imgs_noisy,noise)
        # print(torch.sum(burst_imgs_noisy[middle_img_idx] - burst_imgs[middle_img_idx]))

        # print(cfg.N,cfg.blind,[input_order[x] for x in range(cfg.input_N)])
        if cfg.color_cat:
            stacked_burst = torch.cat([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
        else:
            stacked_burst = torch.stack([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)

        # if cfg.input_noise:
        #     stacked_burst = torch.normal(stacked_burst,noise)

        # -- extract target image --
        if cfg.blind:
            t_img = burst_imgs[middle_img_idx]
        else:
            t_img = szm(raw_img.cuda(non_blocking=True))

        # -- denoising --
        loss,rec_img = model(stacked_burst,middle_img)
        # rec_imgs = rearrange(rec_imgs,'b (n c) h w -> (b n) c h w',n=cfg.input_N-1)
        # loss = 0
        
        # # -- compute loss --
        # r_middle_img = middle_img.repeat(N-1,1,1,1)
        # mse_loss = F.mse_loss(r_middle_img,rec_imgs,reduction='none')
        # # loss += torch.mean(mse_loss)
        # std_est = torch.mean( mse_loss, dim=(1,2,3) )
        # loss += torch.norm(std_est.unsqueeze(1) - std_est)

        # # -- reconstruct image --
        # rec_imgs = rec_imgs.reshape(BS,N-1,C,H,W)
        # # rec_imgs = rearrange( rec_imgs, '(b n) c h w -> b n c h w',n=cfg.input_N-1)
        # rec_img = torch.mean( rec_imgs, dim=1)
        # loss += F.mse_loss( rec_img, middle_img)

        
        # loss = F.mse_loss(t_img,rec_img)

        # -- dncnn denoising --
        # rec_res = model(stacked_burst)

        # -- compute loss --
        # t_res = t_img - burst_imgs[middle_img_idx]
        # loss = F.mse_loss(t_res,rec_res)

        # -- update info --
        running_loss += loss.item()
        total_loss += loss.item()

        # -- BP and optimize --
        loss.backward()
        optimizer.step()

        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:

            # -- compute mse for fun --
            BS = raw_img.shape[0]            
            raw_img = raw_img.cuda(non_blocking=True)
            mse_loss = F.mse_loss(raw_img,rec_img+0.5,reduction='none').reshape(BS,-1)
            mse_loss = torch.mean(mse_loss,1).detach().cpu().numpy()
            psnr = np.mean(mse_to_psnr(mse_loss))
            running_loss /= cfg.log_interval
            print("[%d/%d][%d/%d]: %2.3e [PSNR]: %2.3e"%(epoch, cfg.epochs, batch_idx,
                                                         len(train_loader),
                                                         running_loss,psnr))
            running_loss = 0
    total_loss /= len(train_loader)
    return total_loss

def test_loop_offset(cfg,model,criterion,test_loader,epoch):
    model.eval()
    model = model.to(cfg.device)
    total_psnr = 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(test_loader):
        # for batch_idx, (burst_imgs, raw_img) in enumerate(test_loader):
    
            BS = raw_img.shape[0]
            
            # -- selecting input frames --
            input_order = np.arange(cfg.N)
            # print("pre",input_order)
            # if cfg.blind or True:
            middle_img_idx = -1
            if not cfg.input_with_middle_frame:
                middle = cfg.N // 2
                # print(middle)
                middle_img_idx = input_order[middle]
                input_order = np.r_[input_order[:middle],input_order[middle+1:]]
            else:
                # input_order = np.arange(cfg.N)
                middle = len(input_order) // 2
                middle_img_idx = input_order[middle]
                input_order = np.arange(cfg.N)
            
            # -- reshaping of data --
            raw_img = raw_img.cuda(non_blocking=True)
            burst_imgs = burst_imgs.cuda(non_blocking=True)

            if cfg.color_cat:
                stacked_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
            else:
                stacked_burst = torch.stack([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
    
            # -- direct denoising --
            middle_img = burst_imgs[middle_img_idx]
            loss,rec_img = model(stacked_burst,middle_img)
            # rec_imgs = model(stacked_burst)
            # rec_imgs = rearrange( rec_imgs, 'b (n c) h w -> b n c h w',n=cfg.input_N-1)
            # rec_img = torch.mean( rec_imgs, dim=1)
            
            # -- dncnn denoising --
            # rec_res = model(stacked_burst)
            # rec_img = burst_imgs[middle_img_idx] + rec_res
            
            # -- compare with stacked targets --
            rec_img = rescale_noisy_image(rec_img)        
            loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
            loss = torch.mean(loss,1).detach().cpu().numpy()
            psnr = mse_to_psnr(loss)

            total_psnr += np.mean(psnr)
            total_loss += np.mean(loss)

            if (batch_idx % cfg.test_log_interval) == 0:
                root = Path(f"{settings.ROOT_PATH}/output/n2n/offset_out_noise/rec_imgs/N{cfg.N}/e{epoch}")
                if not root.exists(): root.mkdir(parents=True)
                fn = root / Path(f"b{batch_idx}.png")
                nrow = int(np.sqrt(cfg.batch_size))
                rec_img = rec_img.detach().cpu()
                grid_imgs = vutils.make_grid(rec_img, padding=2, normalize=True, nrow=nrow)
                plt.imshow(grid_imgs.permute(1,2,0))
                plt.savefig(fn)
                plt.close('all')

    ave_psnr = total_psnr / len(test_loader)
    ave_loss = total_loss / len(test_loader)
    print("[Blind: %d | N: %d] Testing results: Ave psnr %2.3e Ave loss %2.3e"%(cfg.blind,cfg.N,ave_psnr,ave_loss))
    return ave_psnr


def train_loop_N_half(cfg,model,optimizer,criterion,train_loader,epoch):
    model.train()
    model = model.to(cfg.device)
    N_half = cfg.N//2
    total_loss = 0
    running_loss = 0


    for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(train_loader):

        optimizer.zero_grad()
        model.zero_grad()

        # reshaping of data
        burst_imgs = burst_imgs.cuda(non_blocking=True)
        inputs,targets = burst_imgs[:N_half],burst_imgs[N_half:]
        stacked_inputs = torch.cat([inputs[x] for x in range(N_half)],dim=1)
        # stacked_targets = torch.cat([targets[x] for x in range(N_half)],dim=1)

        # denoising
        rec_img = model(stacked_inputs)

        # compare with stacked targets
        rec_img = rec_img.expand(targets.shape)
        loss = F.mse_loss(targets,rec_img)
        running_loss += loss.item()
        total_loss += loss.item()

        # BP and optimize
        loss.backward()
        optimizer.step()

        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:
            running_loss /= cfg.log_interval
            print("[%d/%d][%d/%d]: %2.3e "%(epoch, cfg.epochs, batch_idx, len(train_loader),
                                            running_loss))
            running_loss = 0
    total_loss /= len(train_loader)
    return total_loss

def test_loop_N_half(cfg,model,criterion,test_loader,epoch):
    model.train()
    model = model.to(cfg.device)
    N_half = cfg.N//2
    running_loss = 0

    for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(test_loader):

        # reshaping of data
        burst_imgs = burst_imgs.cuda(non_blocking=True)
        inputs,targets = burst_imgs[:N_half],burst_imgs[N_half:]
        stacked_inputs = torch.cat([inputs[x] for x in range(N_half)],dim=1)
        # stacked_targets = torch.cat([targets[x] for x in range(N_half)],dim=1)

        # denoising
        rec_img = model(stacked_inputs)
        
        # compare with stacked targets
        rec_img = rec_img.expand(targets.shape)
        loss = F.mse_loss(targets,rec_img)
        running_loss += loss.item()

        # BP and optimize
        loss.backward()
        optimizer.step()

        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:
            running_loss /= cfg.log_interval
            print("[%d/%d][%d/%d]: %2.3e "%(epoch, cfg.epochs, batch_idx, len(train_loader),
                                            running_loss))
            running_loss = 0

