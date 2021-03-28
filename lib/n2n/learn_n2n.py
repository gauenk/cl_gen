
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
from datasets.transforms import get_noise_transform
from pyutils.vst import anscombe,anscombe_nmlz

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
    noise_level = cfg.noise_params['g']['stddev']
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

    cfg.noise_params['qis']['alpha'] = 4.0
    cfg.noise_params['qis']['readout'] = 0.0
    cfg.noise_params['qis']['nbits'] = 3
    noise_xform = get_noise_transform(cfg.noise_params,
                                      use_to_tensor=False)

    for batch_idx, (burst, res_img, raw_img, d) in enumerate(train_loader):

        optimizer.zero_grad()
        model.zero_grad()

        # -- reshaping of data --
        BS = raw_img.shape[0]
        raw_img = raw_img.cuda(non_blocking=True)
        burst = burst.cuda(non_blocking=True)
        
        # -- anscombe --
        if cfg.use_anscombe:
            burst = anscombe_nmlz.forward(cfg,burst+0.5)-0.5
            
        # print_tensor_stats("burst",burst)
        burst0 = burst[[0]]
        burst1 = burst[[1]]
        # img0 = burst[0]
        # img1 = burst[1]
        # kindex_ds = kIndexPermLMDB(cfg.batch_size,cfg.N)
        # kindex = kindex_ds[batch_idx].cuda(non_blocking=True)
        # kindex = None
        # sim_burst = compute_similar_bursts(cfg,burst0,burst1,K,noise_level/255.,
        #                                    patchsize=cfg.sim_patchsize,
        #                                    shuffle_k=cfg.sim_shuffleK,
        #                                    kindex=kindex,only_middle=True,
        #                                    search_method=cfg.sim_method,
        #                                    db_level="frame")

        # 
        # -- select outputs --
        #

        # -- supervised --
        # img0 = burst[0]
        # img1 = get_nmlz_img(cfg,raw_img)
        # if cfg.use_anscombe: img1 = anscombe_nmlz.forward(cfg,img1+0.5)-0.5
            
        # -- noise2noise: mismatch noise --
        # img0 = burst[0]
        # img1 = torch.normal(raw_img-0.5,75./255.)

        # -- noise2noise --
        img0 = burst[0]
        img1 = burst[1]

        # img1 = noise_xform(raw_img)
        # img1 = img1.cuda(non_blocking=True)
        # raw_img = raw_img.cuda(non_blocking=True)
        # if cfg.use_anscombe: img1 = anscombe_nmlz.forward(cfg,img1+0.5)-0.5

        # raw_img = raw_img.cuda(non_blocking=True)
        # tv_utils.save_image(img0,'noisy0.png')
        # tv_utils.save_image(img1,'noisy1.png')
        # img1 = burst[1]

        # -- noise2noise + one-denoising-level --
        # img0 = burst[0]
        # img1 = burst[1]
        # if cfg.global_steps < 1000: img1 = burst[1]
        # else: img1 = model(burst[1]).detach()

        # -- noise2sim --
        # img0 = burst[0]
        # img1 = sim_burst[0][:,0]

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

            # -- anscombe --
            if cfg.use_anscombe:
                rec_img = anscombe_nmlz.backward(cfg,rec_img + 0.5) - 0.5

            # -- qis noise --
            if noise_type == "qis": rec_img = quantize_img(cfg,rec_img+0.5)-0.5

            # -- raw image normalized for noise --
            raw_img = get_nmlz_img(cfg,raw_img)

            # -- psnr finally --
            loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
            loss = torch.mean(loss,1).detach().cpu().numpy()
            psnr = mse_to_psnr(loss)
            psnr_ave = np.mean(psnr)
            psnr_std = np.std(psnr)

            # print( f"Ratio of noisy to clean: {img0.mean().item() / nmlz_raw.mean().item()}" )
            # print_tensor_stats("img1",img1)
            print_tensor_stats("rec_img",rec_img)
            print_tensor_stats("raw_img",raw_img)
            # print_tensor_stats("nmlz_raw",nmlz_raw)
            # tv_utils.save_image(img0,'learn_noisy0.png',nrow=BS,normalize=True)
            # tv_utils.save_image(rec_img,'learn_rec_img.png',nrow=BS,normalize=True)
            # tv_utils.save_image(raw_img,'learn_raw_img.png',nrow=BS,normalize=True)
            # tv_utils.save_image(nmlz_raw,'learn_nmlz_raw.png',nrow=BS,normalize=True)

            running_loss /= cfg.log_interval
            print("[%d/%d][%d/%d]: %2.3e [PSNR] %2.2f +/- %2.2f "%(epoch, cfg.epochs, batch_idx, len(train_loader),running_loss,psnr_ave,psnr_std))
            running_loss = 0
        cfg.global_steps += 1
    total_loss /= len(train_loader)
    return total_loss

def add_color_channel(bw_pic):
    repeat = [1 for i in bw_pic.shape]
    repeat[-3] = 3
    bw_pic = bw_pic.repeat(*(repeat))
    return bw_pic

def get_nmlz_img(cfg,raw_img):
    noise_type = cfg.noise_params.ntype
    if noise_type in ["g","hg"]: nmlz_raw = raw_img - 0.5
    elif noise_type in ["qis"]:

        # -- convert to bw --
        raw_img_bw = tvF.rgb_to_grayscale(raw_img,1)
        raw_img_bw = add_color_channel(raw_img_bw)

        # -- quantize as from adc(poisson_mean) --
        raw_img_bw = quantize_img(cfg,raw_img_bw)

        # -- start dnn normalization for optimization --
        nmlz_raw = raw_img_bw - 0.5
    else:
        print("[Warning]: Check normalize raw image.")        
        nmlz_raw = raw_img
    return nmlz_raw

def quantize_img(cfg,image):

    params = cfg.noise_params['qis']
    pix_max = 2**params['nbits'] - 1

    image *= params['alpha']
    image = torch.round(image)
    image = torch.clamp(image, 0, pix_max)
    image /= params['alpha']

    return image

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

            # -- anscombe --
            if cfg.use_anscombe:
                img0 = anscombe_nmlz.forward(cfg,img0+0.5) - 0.5

            # denoising
            rec_img = model(img0)

            # -- anscombe --
            if cfg.use_anscombe:
                rec_img = anscombe_nmlz.backward(cfg,rec_img + 0.5) - 0.5

            # compare with stacked targets
            # rec_img = rescale_noisy_image(rec_img)
            if noise_type == "qis": rec_img = quantize_img(cfg,rec_img+0.5)-0.5
            nmlz_raw = get_nmlz_img(cfg,raw_img)
            loss = F.mse_loss(nmlz_raw,rec_img,reduction='none').reshape(BS,-1)
            loss = torch.mean(loss,1).detach().cpu().numpy()

            # -- check for perfect matches --
            PSNR_MAX = 50
            if np.any(np.isinf(loss)):
                loss = []
                for b in range(BS):
                    if np.isinf(loss[b]): loss.append(PSNR_MAX)
                    else: loss.append(loss[b])
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

