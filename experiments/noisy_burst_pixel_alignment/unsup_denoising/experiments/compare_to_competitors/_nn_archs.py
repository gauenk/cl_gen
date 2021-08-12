
import torch
from torch import nn as nn
from torch import optim as optim
from einops import rearrange,repeat

from layers.ame_kpn.KPN import KPN as KPN_model,LossFunc
from layers.fastdvdnet.models import FastDVDnet

def get_nn_model(cfg,nn_arch):
    if nn_arch == "kpn":
        return get_kpn_model(cfg)
    elif nn_arch == "fdvd":
        return get_fdvd_model(cfg)        
    else:
        raise ValueError(f"Uknown nn architecture [{nn_arch}]")

def get_kpn_model(cfg):
    model = KPN_model(color=True,burst_length=cfg.nframes,blind_est=True,
                      kernel_size=[5],cascade=False) 
    model = model.to(cfg.gpuid,non_blocking=True)
    loss_fxn_base = LossFunc(tensor_grad=True,alpha=1.0)    
    loss_fxn_base = loss_fxn_base.to(cfg.gpuid,non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -- create closure for loss --
    def wrap_loss_fxn(denoised,gt_img,denoised_frames,step):
        gt_img_nmlz = gt_img - 0.5#gt_img.mean()
        loss_basic,loss_anneal = loss_fxn_base(denoised_frames,denoised,gt_img_nmlz,step)
        return loss_basic + loss_anneal
    loss_fxn = wrap_loss_fxn

    # -- create empty scheduler --
    def scheduler_fxn(epoch):
        pass

    # -- wrap call function for interface --
    forward_fxn = model.forward
    def wrap_forward(dyn_noisy,noise_info):
        noisy = dyn_noisy - 0.5#dyn_noisy.mean()
        kpn_stack = rearrange(noisy,'t b c h w -> b t c h w')
        kpn_cat = rearrange(noisy,'t b c h w -> b (t c) h w')
        denoised,denoised_ave,filters = forward_fxn(kpn_cat,kpn_stack)
        denoised_ave += 0.5
        return denoised_ave,denoised
    model.forward = wrap_forward

    return model,loss_fxn,optimizer,scheduler_fxn

def get_fdvd_model(cfg):
    model = FastDVDnet(cfg.nframes)
    model = model.to(cfg.gpuid,non_blocking=True)
    loss_fxn_base = nn.MSELoss(reduction='sum')
    loss_fxn_base = loss_fxn_base.to(cfg.gpuid,non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -- create loss function closure --
    def wrap_loss_fxn(denoised,gt_img,denoised_frames,step):
        return loss_fxn_base(denoised,gt_img)
    loss_fxn = wrap_loss_fxn

    # -- create scheduler closure --
    def scheduler_fxn(epoch):
        argdict = {'lr':1e-3,'milestone':[50,60]}
        current_lr, reset_orthog = fdvd_lr_scheduler(epoch, argdict)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    # -- wrap call function for interface --
    forward_fxn = model.forward
    def wrap_forward(dyn_noisy,noise_params):
        T,B,C,H,W = dyn_noisy.shape
        burst = rearrange(dyn_noisy,'t b c h w -> b (t c) h w')
        # burst /= burst.max()
        noise_map = noise_params.g.std * torch.ones(B,1,H,W) / 255.
        noise_map = noise_map.to(burst.device,non_blocking=True)
        return forward_fxn(burst,noise_map)
    model.forward = wrap_forward

    return model,loss_fxn,optimizer,scheduler_fxn

def fdvd_lr_scheduler(epoch, argdict):
	"""Returns the learning rate value depending on the actual epoch number
	By default, the training starts with a learning rate equal to 1e-3 (--lr).
	After the number of epochs surpasses the first milestone (--milestone), the
	lr gets divided by 100. Up until this point, the orthogonalization technique
	is performed (--no_orthog to set it off).
	"""
	# Learning rate value scheduling according to argdict['milestone']
	reset_orthog = False
	if epoch > argdict['milestone'][1]:
		current_lr = argdict['lr'] / 1000.
		reset_orthog = True
	elif epoch > argdict['milestone'][0]:
		current_lr = argdict['lr'] / 10.
	else:
		current_lr = argdict['lr']
	return current_lr, reset_orthog
