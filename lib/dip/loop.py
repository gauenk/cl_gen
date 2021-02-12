
# -- python imports --
import bm3d
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange, repeat, reduce
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as th_trans

# -- project code --
import settings
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transform import ScaleZeroMean
from .model_io import load_model_attn,load_model_kpn,load_model_skip
from .optim_io import load_optimizer

def weights_init(m):
    weight_bool = isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d)
    if weight_bool:
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.zero_()

def dip_loop(cfg,test_loader,epoch):

    total_psnr = 0
    total_loss = 0
    num_samples = 0
    ave_psnrs,std_psnrs = [],[]
    for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(test_loader):
    # for batch_idx, (burst_imgs, raw_img) in enumerate(test_loader):

        BS = raw_img.shape[0]
        N,BS,C,H,W = burst_imgs.shape
        
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
        # print("post",input_order,middle_img_idx,cfg.blind,cfg.N)

        
        # -- reshaping of data --
        # raw_img = raw_img.cuda(non_blocking=True)
        # burst_imgs = burst_imgs.cuda(non_blocking=True)

        if cfg.color_cat:
            stacked_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
        else:
            stacked_burst = torch.stack([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)

        # stacked_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=0)
        # stacked_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=0)
        stacked_burst = torch.stack([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
        cat_burst = torch.cat([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)

        # -- dip denoising --
        # img = burst_imgs[middle_img_idx] + 0.5
        t_img = burst_imgs[middle_img_idx] + 0.5
        img = stacked_burst + 0.5

        # -- baseline psnr --
        ave_rec = torch.mean(stacked_burst,dim=1)+0.5
        b_loss = F.mse_loss(raw_img,ave_rec,reduction='none').reshape(BS,-1)
        b_loss = torch.mean(b_loss,1).detach().cpu().numpy()
        ave_psnr = np.mean(mse_to_psnr(b_loss))

        # -- bm3d --
        bm3d_rec = bm3d.bm3d(t_img[0].transpose(0,2), sigma_psd=25/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        bm3d_rec = torch.FloatTensor(bm3d_rec).transpose(0,2)
        b_loss = F.mse_loss(raw_img[0],bm3d_rec,reduction='none').reshape(BS,-1)
        b_loss = torch.mean(b_loss,1).detach().cpu().numpy()
        bm3d_nb_psnr = np.mean(mse_to_psnr(b_loss))

        # -- blind bm3d --
        noisy_mid = t_img[0].transpose(0,2)
        bm3d_rec = None
        if bm3d_rec is None: sigma_est = torch.std(noisy_mid-0.5)
        else: sigma_est = torch.std(noisy_mid - bm3d_rec.transpose(0,2))
        bm3d_rec = bm3d.bm3d(noisy_mid, sigma_psd=sigma_est, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        bm3d_rec = torch.FloatTensor(bm3d_rec).transpose(0,2)
        b_loss = F.mse_loss(raw_img[0],bm3d_rec,reduction='none').reshape(BS,-1)
        b_loss = torch.mean(b_loss,1).detach().cpu().numpy()
        bm3d_b_psnr = np.mean(mse_to_psnr(b_loss))

        # img = torch.normal(raw_img,25./255)
        # z = torch.normal(0,torch.ones_like(img[0].unsqueeze(0)))
        # print(z.shape)
        # z = z.requires_grad_(True)
        diff = 100
        iters = 2000
        tol = 5e-9
        # params = [params.data.clone() for params in model.parameters()]
        # stacked_burst = torch.normal(0,torch.ones( ( BS, N, C, H, W) ))
        # stacked_burst = stacked_burst.cuda(non_blocking=True)
        # cat_burst = rearrange(stacked_burst,'bs n c h w -> bs (n c) h w')
        repeats = 1

        best_attn_psnr,best_kpn_psnr,best_cnn_psnr = 0,0,0
        best_rec = 0
        psnrs = []
        for repeat in range(repeats):
            idx = 0
            repeat_psnr = 0

            attn_info = get_attn_model(cfg,0)
            kpn_info = get_kpn_model(cfg,1)
            cnn_info = get_cnn_model(cfg,2)
            z = torch.normal(0,torch.ones_like(t_img))
            z_stack = torch.normal(0,torch.ones_like(stacked_burst))

            attn_loss_diff,attn_loss_prev = 1.,0
            while (idx < iters):
                idx += 1
                # z_img = z + torch.normal(0,torch.ones_like(z)) * 1./20
                # stacked_burst_i = torch.normal(stacked_burst,1./20)
                # cat_burst_i = torch.normal(cat_burst,1./20)
                # print('m',torch.mean( (stacked_burst_i - stacked_burst)**2) )
                # z_img = z
                # rec_img = model(z_img)
                # -- create inputs for kpn --
    
                # -- attn model -- 
                fwd_args = [attn_info,raw_img,stacked_burst,idx,iters,attn_loss_diff,attn_loss_prev]
                attn_psnr,loss_attn,rec_attn,attn_loss_diff,attn_loss_prev = attn_forward(*fwd_args)
                # attn_psnr,loss_attn,rec_attn,attn_loss_diff,attn_loss_prev = 0,torch.Tensor([0]),0,0,0

                # -- kpn model --
                fwd_args = [kpn_info,cfg,raw_img,stacked_burst,cat_burst,t_img,idx,iters]
                kpn_psnr,loss_kpn,rec_kpn = kpn_forward(*fwd_args)

                # -- cnn model; middle frame -- 
                fwd_args = [cnn_info,cfg,z,raw_img,t_img]
                cnn_psnr,loss_cnn,rec_cnn = cnn_forward(*fwd_args)

                if (idx % 1) == 0 or idx == 1:
                    if (idx % 250) == 0 or idx == 1:
                        print("[%d] [%d/%d] [PSNR] [attn: %2.2f] [kpn: %2.2f] [cnn-m: %2.2f] [ave: %2.2f] [bm3d-nb: %2.2f] [bm3d-b: %2.2f]" % (batch_idx,idx,iters,attn_psnr,kpn_psnr,cnn_psnr,ave_psnr,bm3d_nb_psnr,bm3d_b_psnr))
                    if attn_psnr > best_attn_psnr:
                        best_attn_psnr = attn_psnr
                        best_rec_attn = rec_attn
                    if kpn_psnr > best_kpn_psnr:
                        best_kpn_psnr = kpn_psnr
                        best_rec_kpn = rec_kpn
                    if cnn_psnr > best_cnn_psnr:
                        best_cnn_psnr = cnn_psnr
                        best_rec_cnn = rec_cnn

                if torch.isinf(loss_attn) or torch.isinf(loss_kpn):
                    print("UH OH! inf loss")
                    break

                # b = list(model.parameters())[0].clone()
                # print("EQ?",torch.equal(a.data,b.data))
                # print(torch.mean(a.data - b.data)**2)
                # params_p = [params.data.clone() for params in model.parameters()]
                # diff = np.mean([float(torch.mean((p - p_p)**2).cpu().item()) for p,p_p in zip(params,params_p)])
                # print("diff: {:.2e}".format(diff))
                # params = params_p
                # if best_psnr > 29: break
            # rec_img = model(z)
        print(f"Best PSNR [attn: {best_attn_psnr}] [kpn: {best_kpn_psnr}] [cnn-m: {best_cnn_psnr}]")
        
        # -- compare with stacked targets --
        # rec_img = rescale_noisy_image(rec_img)        
        # loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
        # loss = torch.mean(loss,1).detach().cpu().numpy()
        # psnr = mse_to_psnr(loss)
        # print(np.mean(psnr))

        # total_psnr += best_psnr
        num_samples += 1
        # ave_psnrs.append(np.mean(psnrs))
        # std_psnrs.append(np.std(psnrs))
        # total_loss += np.mean(loss)

        # if (batch_idx % cfg.test_log_interval) == 0:
        if True:
            root = Path(f"{settings.ROOT_PATH}/output/dip/rec_imgs/N{cfg.N}/e{epoch}/")
            if not root.exists(): root.mkdir(parents=True)
            fn = root / Path(f"b{batch_idx}_attn.png")
            nrow = 2 # int(np.sqrt(cfg.batch_size))
            rec_img = best_rec_attn.detach().cpu()
            # rec_img -= rec_img.min()
            # rec_img /= rec_img.max()
            # print(rec_img.mean(),rec_img.min(),rec_img.max())
            # print(raw_img.mean(),raw_img.min(),raw_img.max())
            save_img = torch.cat([rec_img,raw_img.cpu()],dim=0)
            grid_imgs = vutils.make_grid(save_img, padding=2, normalize=False, nrow=nrow, pad_value=0)
            plt.title(f"PSNR: {best_attn_psnr}")
            plt.imshow(grid_imgs.permute(1,2,0))
            plt.savefig(fn)
            plt.close('all')
            print(f"Saved figure to {fn}")

    ave_psnr = total_psnr / num_samples
    # ave_psnr = total_psnr / len(test_loader)
    # ave_loss = total_loss / len(test_loader)
    print("[Blind: %d | N: %d] Testing results: Ave psnr %2.3e Ave loss %2.3e"%(cfg.blind,cfg.N,ave_psnr,ave_loss))
    return ave_psnr


def get_cnn_model(cfg,gpuid):
    
    model,_ = load_model_skip(cfg)
    optimizer = load_optimizer(cfg,model)        
    model = model.cuda(gpuid)
    model.apply(weights_init)

    info = edict()
    info.model = model
    info.optimizer = optimizer
    info.gpuid = gpuid
    return info

def get_attn_model(cfg,gpuid):
    model,_ = load_model_attn(cfg)
    optimizer = load_optimizer(cfg,model)        
    loss_diff,loss_prev = 1.,0
    model = model.cuda(gpuid)
    model.apply(weights_init)

    info = edict()
    info.model = model
    info.optimizer = optimizer
    info.gpuid = gpuid
    return info

def get_kpn_model(cfg,gpuid):
    model,criterion = load_model_kpn(cfg)
    optimizer = load_optimizer(cfg,model)        
    loss_diff,loss_prev = 1.,0
    model = model.cuda(gpuid)
    model.apply(weights_init)
    cfg.global_step = 0

    info = edict()
    info.model = model
    info.criterion = criterion
    info.optimizer = optimizer
    info.gpuid = gpuid
    return info



def cnn_forward(cnn_info,cfg,z,raw_img,t_img):

    # -- cuda --
    gpuid = cnn_info.gpuid
    raw_img = raw_img.cuda(gpuid)
    cnn_info.model = cnn_info.model.cuda(gpuid)
    z = z.cuda(gpuid)
    raw_img = raw_img.cuda(gpuid)
    t_img=t_img.cuda(gpuid)

    # -- init --
    BS = raw_img.shape[0]
    cnn_info.optimizer.zero_grad()
    cnn_info.model.zero_grad()

    # -- forward pass --
    z_prime = torch.normal(z,1./20)
    rec_img = cnn_info.model(z_prime)
    loss = F.mse_loss(t_img,rec_img)

    # -- sgd step --
    loss.backward()
    cnn_info.optimizer.step()
    
    # -- psnr --
    loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
    loss = torch.mean(loss,1).detach().cpu().numpy()
    psnr = np.mean(mse_to_psnr(loss))

    return psnr,loss,rec_img


def kpn_forward(kpn_info,cfg,raw_img,stacked_burst,cat_burst,t_img,idx,iters):

    # -- cuda --
    gpuid = kpn_info.gpuid
    raw_img = raw_img.cuda(gpuid)
    kpn_info.model = kpn_info.model.cuda(gpuid)
    stacked_burst=stacked_burst.cuda(gpuid)
    cat_burst=cat_burst.cuda(gpuid)
    t_img=t_img.cuda(gpuid)

    # -- init --
    BS = raw_img.shape[0]
    kpn_info.optimizer.zero_grad()
    kpn_info.model.zero_grad()

    # -- forward pass --
    rec_img_i,rec_img = kpn_info.model(cat_burst,stacked_burst)
    lossE_ = kpn_info.criterion(rec_img_i, rec_img, t_img, cfg.global_step)
    # lossE_ = criterion(rec_img_i, rec_img, t_img, cfg.global_step)
    lossE = np.sum(lossE_)

    # -- sgd step --
    lossE.backward()
    kpn_info.optimizer.step()

    # -- post --
    cfg.global_step += 30
    # rec_img += 0.5
    
    # -- psnr --
    loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
    loss = torch.mean(loss,1).detach().cpu().numpy()
    psnr = np.mean(mse_to_psnr(loss))

    return psnr,lossE,rec_img


def attn_forward(attn_info,raw_img,stacked_burst,idx,iters,loss_diff,loss_prev):

    # -- cuda --
    gpuid = attn_info.gpuid
    stacked_burst = stacked_burst.cuda(gpuid)
    raw_img = raw_img.cuda(gpuid)
    attn_info.model = attn_info.model.cuda(gpuid)
    
    # -- init --
    BS = raw_img.shape[0]
    attn_info.optimizer.zero_grad()
    attn_info.model.zero_grad()

    # -- forward pass --
    lossE,rec_img = attn_info.model(stacked_burst,stacked_burst,idx,iters,loss_diff)
    loss_diff = lossE.item() - loss_prev
    loss_prev = lossE.item()

    # -- sgd step --
    lossE.backward()
    attn_info.optimizer.step()

    # -- psnr --
    rec_img += 0.5
    loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
    loss = torch.mean(loss,1).detach().cpu().numpy()
    psnr = np.mean(mse_to_psnr(loss))

    return psnr,lossE,rec_img,loss_diff,loss_prev
