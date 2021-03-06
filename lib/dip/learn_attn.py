
# -- python imports --
import numpy as np
import numpy.random as npr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange, repeat, reduce

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
from .model_io import load_model_attn
from .optim_io import load_optimizer

# -- 3d party imports --
from performer_pytorch import Performer,PerformerLM,FastAttention

def weights_init(m):
    # if isinstance(m, nn.Conv2d):
    #     torch.nn.init.xavier_uniform_(m.weight.data)
    weight_bool = isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d)
    # weight_bool = weight_bool or isinstance(m, Performer) or isinstance(m, nn.Transformer)
    if weight_bool:
        nn.init.kaiming_normal_(m.weight.data)
        # if m.bias is None: print(m)
        m.bias.data.zero_()
        # if not m.bias is None: m.bias.data.zero_()
    # if hasattr(m, 'weight'):
    #     if isinstance(m,LayerNorm): return
    #     print(m)
    #     torch.nn.init.xavier_uniform_(m.weight.data)


def data_loop(cfg,model,optimizer,criterion,test_loader,epoch):

    model.train()
    model = model.to(cfg.device)
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
        raw_img = raw_img.cuda(non_blocking=True)
        burst_imgs = burst_imgs.cuda(non_blocking=True)

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
        # img = torch.normal(raw_img,25./255)
        # z = torch.normal(0,torch.ones_like(img[0].unsqueeze(0)))
        # print(z.shape)
        # z = z.requires_grad_(True)
        diff = 100
        iters = 50
        tol = 5e-9
        # params = [params.data.clone() for params in model.parameters()]
        # stacked_burst = torch.normal(0,torch.ones( ( BS, N, C, H, W) ))
        # stacked_burst = stacked_burst.cuda(non_blocking=True)
        # cat_burst = rearrange(stacked_burst,'bs n c h w -> bs (n c) h w')
        repeats = 1

        best_psnr = 0
        psnrs = []
        for repeat in range(repeats):
            idx = 0
            repeat_psnr = 0
            model,_ = load_model_attn(cfg)
            optimizer = load_optimizer(cfg,model)        
            model = model.cuda()
            model.apply(weights_init)
            # print(f"global_step: {cfg.global_step}")
            cfg.global_step = 0
            z = torch.normal(0,torch.ones_like(stacked_burst))
            while (idx < iters):
                idx += 1
                optimizer.zero_grad()
                model.zero_grad()
                # z_img = z + torch.normal(0,torch.ones_like(z)) * 1./20
                # stacked_burst_i = torch.normal(stacked_burst,1./20)
                # cat_burst_i = torch.normal(cat_burst,1./20)
                # print('m',torch.mean( (stacked_burst_i - stacked_burst)**2) )
                # z_img = z
                # rec_img = model(z_img)
                # -- create inputs for kpn --
                stacked_burst = torch.stack([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
                # cat_burst = torch.cat([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
    
                # -- forward attn model -- 
                z_img = torch.normal(z,torch.ones_like(z) * 1./20)
                # lossE,rec_img = model(z_img,stacked_burst)
                lossE,rec_img = model(stacked_burst,stacked_burst)
                rec_img += 0.5
                # lossE,rec_img = model(stacked_burst,t_img)
                
                # lossE_ = criterion(rec_img_i, rec_img, t_img, cfg.global_step)
                # lossE_ = criterion(rec_img_i, rec_img, t_img, cfg.global_step)
                # cfg.global_step += 30
                # lossE = np.sum(lossE_)
                # lossE = F.mse_loss(t_img,rec_img)
                # lossE = np.sum([F.mse_loss(t_img,rec_img_i[:,i]) for i in range(N)])
    
                # lossE = F.mse_loss(t_img,rec_img)
                # print(idx,lossE)
                if (idx % 1) == 0 or idx == 1:
                    loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
                    loss = torch.mean(loss,1).detach().cpu().numpy()
                    psnr = np.mean(mse_to_psnr(loss))
                    if (idx % 5) == 0 or idx == 1:
                        print("[%d/%d] lossE: [%.2e] psnr: [%.2f]" % (idx,iters,lossE,psnr))
                    if psnr > best_psnr: best_psnr = psnr
                    if psnr > repeat_psnr: repeat_psnr = psnr
                if torch.isinf(lossE): break
                # a = list(model.parameters())[0].clone()
                lossE.backward()
                optimizer.step()
                # b = list(model.parameters())[0].clone()
                # print("EQ?",torch.equal(a.data,b.data))
                # print(torch.mean(a.data - b.data)**2)
                # params_p = [params.data.clone() for params in model.parameters()]
                # diff = np.mean([float(torch.mean((p - p_p)**2).cpu().item()) for p,p_p in zip(params,params_p)])
                # print("diff: {:.2e}".format(diff))
                # params = params_p
            # rec_img = model(z)
            psnrs.append(repeat_psnr)
        print(f"Best PSNR: {best_psnr}")
        
        # -- compare with stacked targets --
        # rec_img = rescale_noisy_image(rec_img)        
        # loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
        # loss = torch.mean(loss,1).detach().cpu().numpy()
        # psnr = mse_to_psnr(loss)
        # print(np.mean(psnr))

        total_psnr += best_psnr
        num_samples += 1
        ave_psnrs.append(np.mean(psnrs))
        std_psnrs.append(np.std(psnrs))
        # total_loss += np.mean(loss)

        # if (batch_idx % cfg.test_log_interval) == 0:
        if True:
            root = Path(f"{settings.ROOT_PATH}/output/dip/rec_imgs/N{cfg.N}/e{epoch}/")
            if not root.exists(): root.mkdir(parents=True)
            fn = root / Path(f"b{batch_idx}.png")
            nrow = 2 # int(np.sqrt(cfg.batch_size))
            rec_img = rec_img.detach().cpu()
            save_img = torch.cat([rec_img,raw_img],dim=0)
            grid_imgs = vutils.make_grid(save_img, padding=0, normalize=True, nrow=nrow)
            plt.imshow(grid_imgs.permute(1,2,0))
            plt.savefig(fn)
            plt.close('all')
            print(f"Saved figure to {fn}")

    ave_psnr = total_psnr / num_samples
    # ave_psnr = total_psnr / len(test_loader)
    # ave_loss = total_loss / len(test_loader)
    print("[Blind: %d | N: %d] Testing results: Ave psnr %2.3e Ave loss %2.3e"%(cfg.blind,cfg.N,ave_psnr,ave_loss))
    return ave_psnr



def train_loop(cfg,model,optimizer,criterion,train_loader,epoch):
    model.train()
    model = model.to(cfg.device)
    N = cfg.N
    total_loss = 0
    running_loss = 0
    szm = ScaleZeroMean()

    for batch_idx, (burst_imgs, res_imgs, raw_img) in enumerate(train_loader):

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
            input_order = np.r_[input_order[:middle],input_order[middle+1:]]
        else:
            middle = len(input_order) // 2
            middle_img_idx = input_order[middle]
            input_order = np.arange(cfg.N)
        # print("post",input_order,middle_img_idx,cfg.blind,cfg.N)

        # -- add input noise --
        burst_imgs = burst_imgs.cuda(non_blocking=True)
        burst_imgs_noisy = burst_imgs.clone()
        if cfg.input_noise:
            noise = np.random.rand() * cfg.input_noise_level
            burst_imgs_noisy[middle_img_idx] = torch.normal(burst_imgs_noisy[middle_img_idx],noise)

        # print(cfg.N,cfg.blind,[input_order[x] for x in range(cfg.input_N)])
        if cfg.color_cat:
            stacked_burst = torch.cat([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
        else:
            stacked_burst = torch.stack([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
        # print("stacked_burst",stacked_burst.shape)

        # if cfg.input_noise:
        #     stacked_burst = torch.normal(stacked_burst,noise)

        # -- extract target image --
        if cfg.blind:
            t_img = burst_imgs[middle_img_idx]
        else:
            t_img = szm(raw_img.cuda(non_blocking=True))

        # -- denoising --
        rec_img = model(stacked_burst)

        # -- compute loss --
        loss = F.mse_loss(t_img,rec_img)

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

    total_loss /= len(train_loader)
    return total_loss


def test_loop(cfg,model,optimizer,criterion,test_loader,epoch):

    model.train()
    model = model.to(cfg.device)
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
        raw_img = raw_img.cuda(non_blocking=True)
        burst_imgs = burst_imgs.cuda(non_blocking=True)

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
        baseline_rec = torch.mean(stacked_burst,dim=1)+0.5
        b_loss = F.mse_loss(raw_img,baseline_rec,reduction='none').reshape(BS,-1)
        b_loss = torch.mean(b_loss,1).detach().cpu().numpy()
        baseline_psnr = np.mean(mse_to_psnr(b_loss))

        # img = torch.normal(raw_img,25./255)
        # z = torch.normal(0,torch.ones_like(img[0].unsqueeze(0)))
        # print(z.shape)
        # z = z.requires_grad_(True)
        diff = 100
        iters = 1000
        tol = 5e-9
        # params = [params.data.clone() for params in model.parameters()]
        # stacked_burst = torch.normal(0,torch.ones( ( BS, N, C, H, W) ))
        # stacked_burst = stacked_burst.cuda(non_blocking=True)
        # cat_burst = rearrange(stacked_burst,'bs n c h w -> bs (n c) h w')
        repeats = 2

        best_psnr = 0
        best_rec = 0
        psnrs = []
        for repeat in range(repeats):
            idx = 0
            repeat_psnr = 0
            model,_ = load_model_attn(cfg)
            optimizer = load_optimizer(cfg,model)        
            loss_diff,loss_prev = 1.,0
            if True:
                model = model.cuda()
                model.apply(weights_init)
                cfg.global_step = 0
                # print(f"global_step: {cfg.global_step}")
            z = torch.normal(0,torch.ones_like(stacked_burst))
            while (idx < iters):
                idx += 1
                optimizer.zero_grad()
                model.zero_grad()
                # z_img = z + torch.normal(0,torch.ones_like(z)) * 1./20
                # stacked_burst_i = torch.normal(stacked_burst,1./20)
                # cat_burst_i = torch.normal(cat_burst,1./20)
                # print('m',torch.mean( (stacked_burst_i - stacked_burst)**2) )
                # z_img = z
                # rec_img = model(z_img)
                # -- create inputs for kpn --
                stacked_burst = torch.stack([burst_imgs[input_order[x]] for x in range(cfg.input_N)],dim=1)
                # cat_burst = torch.cat([burst_imgs_noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
    
                # -- forward attn model -- 
                # z_img = torch.normal(z,torch.ones_like(z) * 1./20)
                # lossE,rec_img = model(z_img,stacked_burst)
                lossE,rec_img = model(stacked_burst,stacked_burst,idx,iters,loss_diff)
                loss_diff = lossE.item() - loss_prev
                loss_prev = lossE.item()

                rec_img += 0.5
                # lossE,rec_img = model(stacked_burst,t_img)

                
                # lossE_ = criterion(rec_img_i, rec_img, t_img, cfg.global_step)
                # lossE_ = criterion(rec_img_i, rec_img, t_img, cfg.global_step)
                # cfg.global_step += 30
                # lossE = np.sum(lossE_)
                # lossE = F.mse_loss(t_img,rec_img)
                # lossE = np.sum([F.mse_loss(t_img,rec_img_i[:,i]) for i in range(N)])
    
                # lossE = F.mse_loss(t_img,rec_img)
                # print(idx,lossE)
                if (idx % 1) == 0 or idx == 1:
                    loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
                    loss = torch.mean(loss,1).detach().cpu().numpy()
                    psnr = np.mean(mse_to_psnr(loss))
                    if (idx % 250) == 0 or idx == 1:
                        print("[%d] [%d/%d] lossE: [%.2e] psnr: [%.2f] baseline psnr: [%.2f]" % (batch_idx,idx,iters,lossE,psnr,baseline_psnr))
                    if psnr > repeat_psnr: repeat_psnr = psnr
                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_rec = rec_img
                if torch.isinf(lossE): break
                # a = list(model.parameters())[0].clone()
                lossE.backward()
                optimizer.step()
                # b = list(model.parameters())[0].clone()
                # print("EQ?",torch.equal(a.data,b.data))
                # print(torch.mean(a.data - b.data)**2)
                # params_p = [params.data.clone() for params in model.parameters()]
                # diff = np.mean([float(torch.mean((p - p_p)**2).cpu().item()) for p,p_p in zip(params,params_p)])
                # print("diff: {:.2e}".format(diff))
                # params = params_p
                # if best_psnr > 29: break
            # rec_img = model(z)
            psnrs.append(repeat_psnr)
        print(f"Best PSNR: {best_psnr}")
        
        # -- compare with stacked targets --
        # rec_img = rescale_noisy_image(rec_img)        
        # loss = F.mse_loss(raw_img,rec_img,reduction='none').reshape(BS,-1)
        # loss = torch.mean(loss,1).detach().cpu().numpy()
        # psnr = mse_to_psnr(loss)
        # print(np.mean(psnr))

        total_psnr += best_psnr
        num_samples += 1
        ave_psnrs.append(np.mean(psnrs))
        std_psnrs.append(np.std(psnrs))
        # total_loss += np.mean(loss)

        # if (batch_idx % cfg.test_log_interval) == 0:
        if True:
            root = Path(f"{settings.ROOT_PATH}/output/dip/rec_imgs/N{cfg.N}/e{epoch}/")
            if not root.exists(): root.mkdir(parents=True)
            fn = root / Path(f"b{batch_idx}.png")
            nrow = 2 # int(np.sqrt(cfg.batch_size))
            rec_img = best_rec.detach().cpu()
            # rec_img -= rec_img.min()
            # rec_img /= rec_img.max()
            # print(rec_img.mean(),rec_img.min(),rec_img.max())
            # print(raw_img.mean(),raw_img.min(),raw_img.max())
            save_img = torch.cat([rec_img,raw_img.cpu()],dim=0)
            grid_imgs = vutils.make_grid(save_img, padding=2, normalize=False, nrow=nrow, pad_value=0)
            plt.title(f"PSNR: {best_psnr}")
            plt.imshow(grid_imgs.permute(1,2,0))
            plt.savefig(fn)
            plt.close('all')
            print(f"Saved figure to {fn}")

    ave_psnr = total_psnr / num_samples
    # ave_psnr = total_psnr / len(test_loader)
    # ave_loss = total_loss / len(test_loader)
    print("[Blind: %d | N: %d] Testing results: Ave psnr %2.3e Ave loss %2.3e"%(cfg.blind,cfg.N,ave_psnr,ave_loss))
    return ave_psnr

