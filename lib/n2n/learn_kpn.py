
# python imports
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# pytorch imports
import torch
import torch.nn.functional as F
import torchvision.utils as vutils


# project code
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

    # if cfg.N != 5: return
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

        # -- reshaping of data --
        # raw_img = raw_img.cuda(non_blocking=True)
        input_order = np.arange(cfg.N)
        # print("pre",input_order,cfg.blind,cfg.N)
        middle_img_idx = -1
        if cfg.blind or True:
            middle = len(input_order) // 2
            # print(middle)
            middle_img_idx = input_order[middle]
            # input_order = np.r_[input_order[:middle],input_order[middle+1:]]
        else:
            input_order = np.arange(cfg.N)
        # print("post",input_order,cfg.blind,cfg.N,middle_img_idx)

        burst_imgs = burst_imgs.cuda(non_blocking=True)
        # print(cfg.N,cfg.blind,[input_order[x] for x in range(cfg.input_N)])
        # stacked_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
        # print("stacked_burst",stacked_burst.shape)
        stacked_burst = torch.stack([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
        cat_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
        # print("burst_imgs.shape",burst_imgs.shape)
        # print("stacked_burst.shape",stacked_burst.shape)
        # -- extract target image --
        if cfg.blind:
            t_img = burst_imgs[middle_img_idx]
        else:
            t_img = szm(raw_img.cuda(non_blocking=True))
        
        # -- denoising --
        rec_img_i,rec_img = model(cat_burst,stacked_burst)
        # print("---")
        # print(rec_img[0].shape)
        # print(rec_img[1].shape)
        # print(torch.mean(torch.sum( torch.pow(rec_img[0] - rec_img[1],2) ) ) )
        # print("^^^^^")
        # rec_img = burst_imgs[middle_img_idx] - rec_res

        # -- compare with stacked burst --
        # print(cfg.blind,t_img.min(),t_img.max(),t_img.mean())
        # rec_img = rec_img.expand(t_img.shape)
        # loss = F.mse_loss(t_img,rec_img)


        # -- compute loss to optimize --
        loss = criterion(rec_img_i, rec_img, t_img, cfg.global_step)
        loss = np.sum(loss)

        # -- update info --
        running_loss += loss.item()
        total_loss += loss.item()

        # -- BP and optimize --
        loss.backward()
        optimizer.step()

        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:
            running_loss /= cfg.log_interval
            print("[%d/%d][%d/%d]: %2.3e "%(epoch, cfg.epochs, batch_idx, len(train_loader),
                                            running_loss))
            running_loss = 0
        cfg.global_step += 1
    total_loss /= len(train_loader)
    return total_loss

def test_loop_offset(cfg,model,criterion,test_loader,epoch):
    model.eval()
    model = model.to(cfg.device)
    total_psnr = 0
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(test_loader):
    
            BS = raw_img.shape[0]
            
            # -- selecting input frames --
            input_order = np.arange(cfg.N)
            # print("pre",input_order)
            if cfg.blind or True:
                middle = cfg.N // 2
                # print(middle)
                middle_img_idx = input_order[middle]
                # input_order = np.r_[input_order[:middle],input_order[middle+1:]]
            else:
                input_order = np.arange(cfg.N)
            
            # reshaping of data
            raw_img = raw_img.cuda(non_blocking=True)
            burst_imgs = burst_imgs.cuda(non_blocking=True)
            stacked_burst = torch.stack([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
            cat_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
    
            # denoising
            rec_img = model(cat_burst,stacked_burst)[1]

            # rec_img = burst_imgs[middle_img_idx] - rec_res
            
            # -- compare with stacked targets --
            rec_img = rescale_noisy_image(rec_img)        

            # -- compute psnr --
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
