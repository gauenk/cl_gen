
# python imports
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange, repeat, reduce

# pytorch imports
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import torchvision.utils as tv_utils

# project code
import settings
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from n2sim.sim_search import compute_similar_bursts,kIndexPermLMDB

def print_tensor_stats(prefix,tensor):
    stats_fmt = (tensor.min().item(),tensor.max().item(),tensor.mean().item())
    stats_str = "%2.2e,%2.2e,%2.2e" % stats_fmt
    print(prefix,stats_str)

def train_loop_n2n(cfg,model,optimizer,criterion,train_loader,epoch):
    model.train()
    model = model.to(cfg.device)
    N = cfg.N
    total_loss = 0
    running_loss = 0
    train_iter = iter(train_loader)
    K = cfg.sim_K
    noise_type = cfg.noise_params.ntype
    noise_level = noise_level = cfg.noise_params['g']['stddev']
    # raw_offset,raw_scale = 0,0
    # if noise_type in ["g","hg"]:
    #     raw_offset = 0.5
    #     if noise_type == "g":
    #         noise_level = cfg.noise_params[noise_type]['stddev']
    #     elif noise_type == "hg":
    #         noise_level = cfg.noise_params[noise_type]['read']
    # elif noise_type == "qis":
    #     noise_params = cfg.noise_params[noise_type]
    #     noise_level = noise_params['readout']
    #     raw_scale = ( 2**noise_params['nbits']-1 ) / noise_params['alpha']

    for batch_idx, (burst, res_img, raw_img, d) in enumerate(train_loader):

        optimizer.zero_grad()
        model.zero_grad()

        # -- reshaping of data --
        BS = raw_img.shape[0]
        raw_img = raw_img.cuda(non_blocking=True)
        burst = burst.cuda(non_blocking=True)
        burst0 = burst[[0]]
        burst1 = burst[[1]]
        # img0 = burst[0]
        # img1 = burst[1]
        # kindex_ds = kIndexPermLMDB(cfg.batch_size,cfg.N)
        # kindex = kindex_ds[batch_idx].cuda(non_blocking=True)
        kindex = None
        sim_burst = compute_similar_bursts(cfg,burst0,burst1,K,noise_level/255.,
                                           patchsize=cfg.sim_patchsize,
                                           shuffle_k=cfg.sim_shuffleK,
                                           kindex=kindex,only_middle=True,
                                           search_method=cfg.sim_method,
                                           db_level="frame")
        # -- select outputs --
        img0 = burst[0]
        img1 = sim_burst[0][:,0]

        # img0 = sim_burst[0][:,0]
        # img1 = sim_burst[0][:,1]
        
        # -- plot example input/output --
        # plt_burst = rearrange(burst,'n b c h w -> (n b) c h w')
        # tv_utils.save_image(plt_burst,'burst.png',nrow=BS,normalize=True)

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
            nmlz_raw = get_nmlz_img(cfg,raw_img)
            loss = F.mse_loss(nmlz_raw,rec_img,reduction='none').reshape(BS,-1)
            loss = torch.mean(loss,1).detach().cpu().numpy()
            psnr = mse_to_psnr(loss)
            psnr_ave = np.mean(psnr)
            psnr_std = np.std(psnr)

            # print( f"Ratio of noisy to clean: {img0.mean().item() / nmlz_raw.mean().item()}" )
            # print_tensor_stats("img1",img1)
            # print_tensor_stats("rec_img",rec_img)
            # print_tensor_stats("nmlz_raw",nmlz_raw)
            tv_utils.save_image(img0,'learn_noisy0.png',nrow=BS,normalize=True)
            tv_utils.save_image(rec_img,'learn_rec_img.png',nrow=BS,normalize=True)
            tv_utils.save_image(raw_img,'learn_raw_img.png',nrow=BS,normalize=True)
            tv_utils.save_image(nmlz_raw,'learn_nmlz_raw.png',nrow=BS,normalize=True)

            running_loss /= cfg.log_interval
            print("[%d/%d][%d/%d]: %2.3e [PSNR] %2.2f +/- %2.2f "%(epoch, cfg.epochs, batch_idx, len(train_loader),running_loss,psnr_ave,psnr_std))
            running_loss = 0
    total_loss /= len(train_loader)
    return total_loss

def add_color_channel(bw_pic):
    repeat = [1 for i in bw_pic.shape]
    repeat[-3] = 3
    bw_pic = bw_pic.repeat(*(repeat))
    return bw_pic

def get_nmlz_img(cfg,raw_img):
    pix_max = 2**3-1
    noise_type = cfg.noise_params.ntype
    if noise_type in ["g","hg"]: nmlz_raw = raw_img - raw_offset
    elif noise_type in ["qis"]:
        params = cfg.noise_params[noise_type]
        pix_max = 2**params['nbits'] - 1
        raw_img_bw = tvF.rgb_to_grayscale(raw_img,1)
        raw_img_bw = add_color_channel(raw_img_bw)
        # nmlz_raw = raw_scale * raw_img_bw - 0.5
        raw_img_bw *= params['alpha']
        raw_img_bw = torch.round(raw_img_bw)
        # print("ll",ll_pic.min().item(),ll_pic.max().item())
        raw_img_bw = torch.clamp(raw_img_bw, 0, pix_max)
        raw_img_bw /= params['alpha']
        # -- end of qis noise transform --

        # -- start dnn normalization for optimization --
        nmlz_raw = raw_img_bw - 0.5
    else:
        print("[Warning]: Check normalize raw image.")        
        nmlz_raw = raw_img
    return nmlz_raw

def test_loop_n2n(cfg,model,criterion,test_loader,epoch):
    model.eval()
    model = model.to(cfg.device)
    total_psnr = 0
    total_loss = 0

    noise_type = cfg.noise_params.ntype
    # raw_offset,raw_scale = 0,0
    # if noise_type in ["g","hg"]:
    #     noise_level = cfg.noise_params[noise_type]['stddev']
    #     raw_offset = 0.5
    # elif noise_type == "qis":
    #     params = cfg.noise_params[noise_type]
    #     noise_level = params['readout']
    #     raw_scale = ( 2**params['nbits']-1 ) / params['alpha']

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
            # rec_img = rescale_noisy_image(rec_img)
            nmlz_raw = get_nmlz_img(cfg,raw_img)
            loss = F.mse_loss(nmlz_raw,rec_img,reduction='none').reshape(BS,-1)
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
                grid_imgs = tv_utils.make_grid(rec_img, padding=2, normalize=True, nrow=nrow)
                plt.imshow(grid_imgs.permute(1,2,0))
                plt.savefig(fn)
                plt.close('all')
  

    ave_psnr = total_psnr / len(test_loader)
    ave_loss = total_loss / len(test_loader)
    print("Testing results: Ave psnr %2.3e Ave loss %2.3e"%(ave_psnr,ave_loss))
    return ave_psnr

