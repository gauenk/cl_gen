
# python imports
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt

# pytorch imports
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

# project code
import settings
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from n2sim.sim_search import compute_similar_bursts,kIndexPermLMDB

def train_loop_n2n(cfg,model,optimizer,criterion,train_loader,epoch):
    model.train()
    model = model.to(cfg.device)
    N = cfg.N
    total_loss = 0
    running_loss = 0
    train_iter = iter(train_loader)
    K = cfg.sim_K
    noise_level = cfg.noise_params['g']['stddev']

    for batch_idx, (burst, res_img, raw_img, d) in enumerate(train_loader):

        optimizer.zero_grad()
        model.zero_grad()

        # -- reshaping of data --
        BS = raw_img.shape[0]
        raw_img = raw_img.cuda(non_blocking=True)
        burst = burst.cuda(non_blocking=True)
        burst0 = burst[[0]]
        # img0 = burst[0]
        # img1 = burst[1]
        # kindex_ds = kIndexPermLMDB(cfg.batch_size,cfg.N)
        # kindex = kindex_ds[batch_idx].cuda(non_blocking=True)
        kindex = None
        sim_burst = compute_similar_bursts(cfg,burst0,K,noise_level/255.,
                                           patchsize=cfg.patchsize,shuffle_k=cfg.sim_shuffleK,
                                           kindex=kindex,only_middle=True,
                                           search_method=cfg.sim_method)
        img0 = sim_burst[0][:,0]
        img1 = sim_burst[0][:,1]
        
        # -- denoising --
        rec_img = model(img0)

        # -- compare with stacked burst --
        # loss = F.mse_loss(raw_img,rec_img)
        loss = F.mse_loss(img1,rec_img)

        # -- update info --
        running_loss += loss.item()
        total_loss += loss.item()

        # -- BP and optimize --
        loss.backward()
        optimizer.step()

        if (batch_idx % cfg.log_interval) == 0 and batch_idx > 0:
            loss = F.mse_loss(raw_img-0.5,rec_img,reduction='none').reshape(BS,-1)
            loss = torch.mean(loss,1).detach().cpu().numpy()
            psnr = mse_to_psnr(loss)
            psnr_ave = np.mean(psnr)
            psnr_std = np.std(psnr)

            running_loss /= cfg.log_interval
            print("[%d/%d][%d/%d]: %2.3e [PSNR] %2.2f +/- %2.2f "%(epoch, cfg.epochs, batch_idx, len(train_loader),running_loss,psnr_ave,psnr_std))
            running_loss = 0
    total_loss /= len(train_loader)
    return total_loss


def test_loop_n2n(cfg,model,criterion,test_loader,epoch):
    model.eval()
    model = model.to(cfg.device)
    total_psnr = 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (burst, res_img, raw_img, d) in enumerate(test_loader):

            BS = raw_img.shape[0]

            # reshaping of data
            raw_img = raw_img.cuda(non_blocking=True)
            burst = burst.cuda(non_blocking=True)
            img0 = burst[0]

            # denoising
            rec_img = model(img0)

            # compare with stacked targets
            rec_img = rescale_noisy_image(rec_img)
            loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
            loss = torch.mean(loss,1).detach().cpu().numpy()
            psnr = mse_to_psnr(loss)

            total_psnr += np.mean(psnr)
            total_loss += np.mean(loss)

            if (batch_idx % cfg.test_log_interval) == 0:
                root = Path(f"{settings.ROOT_PATH}/output/n2n/rec_imgs/e{epoch}")
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
    print("Testing results: Ave psnr %2.3e Ave loss %2.3e"%(ave_psnr,ave_loss))
    return ave_psnr

