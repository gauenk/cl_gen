

# -- python imports --
from pathlib import Path
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
from layers.attn import TransformerNetwork32_v5
from layers.ame_kpn.KPN import KPN as KPN_model,LossFunc
from layers.ame_kpn.KPN_1f import KPN_1f,KPN_1f_fs64,KPN_1f_cls_fs64,KPN_1f_cls_fs32
from layers.unet import UNetN_v2,UNet_n2n,UNet_Git,UNet_Git3d
from layers import UNetGP
from layers.burst import BurstAlignN2N,BurstAlignSG,BurstRecLoss,NoiseCriticModel
from layers.denoise_gan import DCGAN_D
from .optim_io import load_optimizer,load_optimizer_gan
from learning.utils import save_model

def load_burst_n2n_model(cfg):

    # -- init --
    init_lr = cfg.init_lr

    # -- load kpn model --
    kpn,_ = load_model_kpn_1f(cfg)

    # -- load unet info --
    unet_info = edict()
    unet_info.model = UNet_n2n( 1,3,3)
    cfg.init_lr = 1e-4
    unet_info.optim = load_optimizer(cfg,unet_info.model)
    unet_info.S = None

    # -- load noise critic info --
    disc_model = DCGAN_D(64, -1,3,64,1,0)
    disc_model = disc_model.cuda(cfg.gpuid)
    cfg.init_lr = 1*1e-4
    sim_params = edict({'mean':0,'std':25./255,'noise_type':'gaussian'})
    disc_optim = load_optimizer_gan(cfg,disc_model)
    noise_critic = NoiseCriticModel(disc_model,disc_optim,sim_params,cfg.device)

    # -- create burstaligned model --
    model = BurstAlignN2N(kpn,unet_info,noise_critic)
    criterion = BurstRecLoss(noise_critic,alpha=1.0)

    # -- model models to cuda --
    model = model.cuda(cfg.gpuid)
    model.unet_info.model = model.unet_info.model.cuda(cfg.gpuid)

    # -- finally --
    cfg.init_lr = init_lr

    return model,criterion

def load_burst_kpn_model(cfg):

    # -- init --
    init_lr = cfg.init_lr

    # -- load kpn model --
    kpn,_ = load_model_kpn_1f_cls(cfg)
    # kpn,_ = load_model_kpn_1f(cfg)

    # -- load unet info --
    denoiser_info = edict()
    denoiser_info.model,_ = load_model_kpn(cfg)
    cfg.init_lr = 1e-4
    denoiser_info.optim = load_optimizer_gan(cfg,denoiser_info.model)
    denoiser_info.S = None

    # -- load noise critic info --
    disc_model = DCGAN_D(64, -1,3,64,1,0)
    disc_model = disc_model.cuda(cfg.gpuid)
    cfg.init_lr = 1*1e-4
    sim_params = edict({'mean':0,'std':25./255,'noise_type':'gaussian'})
    disc_optim = load_optimizer_gan(cfg,disc_model)
    p_lambda = 10
    noise_critic = NoiseCriticModel(disc_model,disc_optim,sim_params,cfg.device,p_lambda)

    # -- create burstaligned model --
    model = BurstAlignSG(kpn,denoiser_info)
    criterion = BurstRecLoss(alpha=1.0)

    # -- model models to cuda --
    model = model.cuda(cfg.gpuid)
    model.denoiser_info.model = model.denoiser_info.model.cuda(cfg.gpuid)

    # -- finally --
    cfg.init_lr = init_lr

    return model,noise_critic,criterion

def load_unet_model(cfg):

    unet = UNet_n2n( cfg.input_N,3,3*(cfg.input_N-1) )
    cfg.color_cat = False

    # model = UNet_Git(3*cfg.input_N,3)
    # cfg.color_cat = True

    # model = UNet_Git3d(cfg.input_N,3)
    # cfg.color_cat = False

    # model = UNetGP(cfg.input_N,cfg.unet_channels)
    # cfg.color_cat = False

    # model = UNetN_v2(cfg.input_N,cfg.unet_channels)
    # model = load_attn_model(cfg,unet)
    model = unet
    return model,None

def load_model_fp(cfg,model,model_fp,rank):
    if cfg.use_ddp:
        map_location = {'cuda:%d' % 0, 'cuda:%d' % rank}
    else:
        map_location = 'cuda:%d' % rank
    print(f"Loading model filepath [{model_fp}]")
    state = torch.load(model_fp, map_location=map_location)
    model.load_state_dict(state)
    return model

def load_model_field(cfg,rank,model,field):
    model = model.to(rank)
    if cfg.load:
        fn = Path("checkpoint_{}.tar".format(cfg.epoch_num))
        model_fp = Path(cfg.model_path) / Path(field) / fn
        model = load_model_fp(cfg,model,model_fp,rank)
    # if cfg.use_ddp:
    #     model = DDP(model, device_ids=[rank])
    return model

def load_model_kpn_1f_cls(cfg):
    if cfg.dynamic.frame_size == 64:
        return KPN_1f_cls_fs64(color=True,kernel_size=[cfg.kpn_frame_size],burst_length=cfg.input_N,blind_est=True),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    elif cfg.dynamic.frame_size == 32:
        return KPN_1f_cls_fs32(color=True,kernel_size=[cfg.kpn_frame_size],burst_length=cfg.input_N,blind_est=True),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    else:
        raise KeyError("Uknown frame size [{cfg.dynamic.frame_size}]")

def load_model_kpn_1f(cfg):
    if cfg.dynamic.frame_size == 128:
        return KPN_1f(color=True,kernel_size=[cfg.kpn_frame_size],burst_length=cfg.input_N,blind_est=True),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    elif cfg.dynamic.frame_size == 64:
        return KPN_1f_fs64(color=True,kernel_size=[cfg.kpn_frame_size],burst_length=cfg.input_N,blind_est=True),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    elif cfg.dynamic.frame_size == 32:
        return KPN_1f_fs32(color=True,kernel_size=[cfg.kpn_frame_size],burst_length=cfg.input_N,blind_est=True),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    else:
        raise KeyError("Uknown frame size [{cfg.dynamic.frame_size}]")


def load_model_kpn(cfg):
    return KPN_model(color=True,burst_length=cfg.input_N,blind_est=True,kernel_size=[cfg.kpn_frame_size]),LossFunc()

def load_attn_model(cfg,unet):
    simclr = None #load_model_simclr(cfg)
    # denoise_model = load_denoise_model_fp(cfg,denoise_model)

    input_patch,output_patch = cfg.patch_sizes
    # d_model = 2048 // (patch_height * patch_width)
    # d_model = 398336 // (patch_height * patch_width)
    d_model = cfg.d_model_attn    
    xformer_args = [simclr, d_model, cfg.dynamic.frame_size, input_patch,
                    output_patch, cfg.input_N, cfg.dataset.bw, unet]
    model = TransformerNetwork32_v5(*xformer_args)
    model = model.to(cfg.device)
    return model

def save_burst_model(cfg, name, model, optimizer=None):
    # -- save model path --
    model_path = Path(cfg.model_path)

    # -- write & save with full info --
    cfg.model_path = model_path / Path("{}/".format(name))
    save_model(cfg, model, optimizer)

    # -- restore model path --
    cfg.model_path = model_path
        
    
