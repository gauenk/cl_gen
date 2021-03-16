

# -- python imports --
from pathlib import Path
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
from torch import nn

# -- project imports --
from layers.attn import TransformerNetwork32_v5
from layers.ame_kpn.KPN import KPN as KPN_model,LossFunc
from layers.ame_kpn.KPN_1f import KPN_1f,KPN_1f_fs64,KPN_1f_cls_fs32,KPN_1f_cls
from layers.unet import UNetN_v2,UNet_n2n,UNet_Git,UNet_Git3d,UNet_n2n_2layer,UNet2
from layers import UNetGP
from layers.burst import BurstAlignN2N,BurstAlignSG,BurstRecLoss,NoiseCriticModel
from layers.stn import STNBurst
from layers.denoise_gan import DCGAN_D
from .optim_io import load_optimizer,load_optimizer_kpn,load_optimizer_gan
from learning.utils import save_model
from layers.csbdeep import CSBDeepBN,init_net

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

    # -- load alignment model --
    align_info = edict()
    align_info.model,_ = load_model_kpn_1f_cls_cascade(cfg)
    align_info.optim = load_optimizer(cfg,align_info.model)
    align_info.S = None

    # -- load denoising model --
    denoiser_info = edict()
    denoiser_info.model,_ = load_model_kpn_cascade(cfg)
    # denoiser_info.model,_ = load_model_kpn(cfg)
    denoiser_info.optim = load_optimizer(cfg,denoiser_info.model)
    denoiser_info.S = None

    # -- load unet model --
    unet_info = edict()
    unet_info.model = load_model_unet(cfg).to(cfg.device)
    # unet_info.model = load_model_unet(cfg).to(cfg.device)
    # unet_info.model = init_net(unet_info.model)
    # unet_info.model,_ = load_model_kpn(cfg)
    unet_info.optim = load_optimizer(cfg,unet_info.model)
    unet_info.S = None

    # -- create burstaligned model --
    use_align = cfg.burst_use_alignment
    use_unet =  cfg.burst_use_unet
    use_unet_only = cfg.burst_use_unet_only
    criterion = BurstRecLoss(alpha=1.0)
    model = BurstAlignSG(align_info,denoiser_info,unet_info,
                         use_alignment=use_align,use_unet=use_unet,
                         use_unet_only=use_unet_only,
                         kpn_num_frames=cfg.kpn_num_frames)

    # -- load noise critic info --
    disc_model = DCGAN_D(64, -1,3,64,1,0)
    disc_model = disc_model.cuda(cfg.gpuid)
    cfg.init_lr = 1*1e-4
    sim_params = edict({'mean':0,'std':25./255,'noise_type':'gaussian'})
    disc_optim = load_optimizer_gan(cfg,disc_model)
    p_lambda = 10
    noise_critic = NoiseCriticModel(disc_model,disc_optim,sim_params,cfg.device,p_lambda)

    # -- model models to cuda --
    model = model.cuda(cfg.gpuid)
    model.denoiser_info.model = model.denoiser_info.model.cuda(cfg.gpuid)
    model.align_info.model = model.align_info.model.cuda(cfg.gpuid)

    # -- finally --
    cfg.init_lr = init_lr

    return model,noise_critic,criterion

def load_burst_stn_model(cfg):

    # -- init --
    init_lr = cfg.init_lr

    # -- load alignment model --
    align_info = edict()
    align_info.model,_ = load_model_stn(cfg)
    align_info.optim = load_optimizer(cfg,align_info.model)
    align_info.S = None

    # -- load denoising model --
    denoiser_info = edict()
    denoiser_info.model,_ = load_model_kpn(cfg)
    denoiser_info.optim = load_optimizer(cfg,denoiser_info.model)
    denoiser_info.S = None

    # -- create burstaligned model --
    model = BurstAlignSTN(align_info,denoiser_info)
    criterion = BurstRecLoss(alpha=1.0)

    # -- load noise critic info --
    disc_model = DCGAN_D(64, -1,3,64,1,0)
    disc_model = disc_model.cuda(cfg.gpuid)
    cfg.init_lr = 1*1e-4
    sim_params = edict({'mean':0,'std':25./255,'noise_type':'gaussian'})
    disc_optim = load_optimizer_gan(cfg,disc_model)
    p_lambda = 10
    noise_critic = NoiseCriticModel(disc_model,disc_optim,sim_params,cfg.device,p_lambda)

    # -- model models to cuda --
    model = model.cuda(cfg.gpuid)
    model.denoiser_info.model = model.denoiser_info.model.cuda(cfg.gpuid)
    model.align_info.model = model.align_info.model.cuda(cfg.gpuid)

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

def load_model_kpn_1f_cls_cascade(cfg):
    if not cfg.kpn_1f_cascade:
        return load_model_kpn_1f_cls(cfg)
    else:
        cascade = []
        for i in range(cfg.kpn_1f_cascade_num):
            not_final = i != (cfg.kpn_1f_cascade_num-1)
            cfg.kpn_1f_cascade_output = not_final
            kpn,loss_fxn = load_model_kpn_1f_cls(cfg)
            cascade.append(kpn)
        cascade = nn.Sequential(*cascade)
        return cascade,loss_fxn

def load_model_kpn_1f_cls(cfg):
    if cfg.dynamic.frame_size in [64,128]:
        return KPN_1f_cls(color=True,kernel_size=[cfg.kpn_1f_frame_size],burst_length=cfg.input_N,blind_est=True,filter_thresh=cfg.kpn_filter_onehot,cascade=cfg.kpn_1f_cascade_output,frame_size=cfg.dynamic.frame_size),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    elif cfg.dynamic.frame_size == 32:
        return KPN_1f_cls_fs32(color=True,kernel_size=[cfg.kpn_1f_frame_size],burst_length=cfg.input_N,blind_est=True,filter_thresh=cfg.kpn_filter_onehot,cascade=cfg.kpn_1f_cascade_output),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    else:
        raise KeyError(f"Uknown frame size [{cfg.dynamic.frame_size}]")

def load_model_kpn_1f(cfg):
    if cfg.dynamic.frame_size == 128:
        return KPN_1f(color=True,kernel_size=[cfg.kpn_1f_frame_size],burst_length=cfg.input_N,blind_est=True),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    elif cfg.dynamic.frame_size == 64:
        return KPN_1f_fs64(color=True,kernel_size=[cfg.kpn_1f_frame_size],burst_length=cfg.input_N,blind_est=True),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    elif cfg.dynamic.frame_size == 32:
        return KPN_1f_fs32(color=True,kernel_size=[cfg.kpn_1f_frame_size],burst_length=cfg.input_N,blind_est=True),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)
    else:
        raise KeyError("Uknown frame size [{cfg.dynamic.frame_size}]")


def load_model_stn(cfg):
    fs = cfg.dynamic.frame_size
    img_size = (3,fs,fs)
    return STNBurst(img_size)


def load_model_kpn_cascade(cfg):
    if not cfg.kpn_cascade:
        return load_model_kpn(cfg)
    else:
        cascade = []
        for i in range(cfg.kpn_cascade_num):
            not_final = i != (cfg.kpn_cascade_num-1)
            cfg.kpn_cascade_output = not_final
            kpn,loss_fxn = load_model_kpn(cfg)
            cascade.append(kpn)
        cascade = nn.Sequential(*cascade)
        return cascade,loss_fxn

def load_model_kpn(cfg):
    return KPN_model(color=True,burst_length=cfg.kpn_num_frames,blind_est=True,kernel_size=[cfg.kpn_frame_size],cascade=cfg.kpn_cascade_output),LossFunc(tensor_grad=~cfg.blind,alpha=1.0)

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

def load_model_unet(cfg):
    # return CSBDeepBN( 3*cfg.N, 3*cfg.N)
    # return UNet_n2n_2layer( 1*cfg.N, 3)
    return UNet2(3*cfg.N,3*cfg.N)
    # return UNet_n2n( cfg.N,3,3*(cfg.N) )
    # return UNet_n2n( 1, 3, 3)

def save_burst_model(cfg, name, model, optimizer=None):
    # -- save model path --
    model_path = Path(cfg.model_path)
    optim_path = Path(cfg.optim_path)

    # -- write & save with full info --
    cfg.model_path = model_path / Path("{}/".format(name))
    cfg.optim_path = optim_path / Path("{}/".format(name))
    save_model(cfg, model, optimizer)

    # -- restore model path --
    cfg.model_path = model_path
    cfg.optim_path = optim_path
        
    
