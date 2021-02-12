
# -- pytorch imports --
import torch

# -- project imports --
from layers.ame_kpn.KPN import KPN as KPN_model,LossFunc
from layers.dip import skip
from layers.unet import UNetN_v2,UNet_n2n,UNet_Git,UNet_Git3d,UNet_small
from layers import UNetGP
from layers.attn import TransformerNetwork32_dip,TransformerNetwork32_dip_v2
from layers.simcl import ClBlockEnc
from simcl.model_io import load_model as simcl_load_model

def load_model_skip(cfg):
    # model = UNet_n2n(cfg.input_N,5)
    # cfg.color_cat = True

    # model = UNet_Git(3*cfg.input_N,3)
    # cfg.color_cat = True

    cfg.color_cat = True
    pad = 'reflection'
    # model = skip(3, 3, 
    #            num_channels_down = [8, 16, 32, 64, 128], 
    #            num_channels_up   = [8, 16, 32, 64, 128],
    #            num_channels_skip = [0, 0, 0, 4, 4], 
    #            upsample_mode='bilinear',
    #            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    model = skip(3, 3, 
                 num_channels_down = [8, 16, 32],
                 num_channels_up   = [8, 16, 32],
                 num_channels_skip = [0, 4, 4],
                 upsample_mode='bilinear',
                 need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    # model = UNetGP(cfg.input_N,cfg.unet_channels)
    # cfg.color_cat = False
    return model,None

def load_model_kpn(cfg):
    return KPN_model(color=True,burst_length=cfg.input_N,blind_est=True),LossFunc()

def load_model_attn(cfg):
    simclr = None #load_model_simclr(cfg)
    d_model_attn = cfg.d_model_attn

    input_patch,output_patch = cfg.patch_sizes
    denoise_model = UNet_small(d_model_attn*cfg.input_N)
    # denoise_model = load_denoise_model_fp(cfg,denoise_model)

    # d_model = 2048 // (patch_height * patch_width)
    # d_model = 398336 // (patch_height * patch_width)
    d_model = d_model_attn    
    xformer_args = [simclr, d_model, cfg.dynamic.frame_size, input_patch,
                    output_patch, cfg.input_N, cfg.dataset.bw, denoise_model, cfg.batch_size]
    model = TransformerNetwork32_dip_v2(*xformer_args)
    model = model.to(cfg.device)
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


def load_model_simclr(cfg):

    # encoder = get_resnet("resnet50", cfg.dataset.name, False, False)
    # projector = Projector(2048, 64)
    # block = ClBlock(encoder,projector,cfg.device,cfg.N,cfg.batch_size)
    # block = block.to(cfg.device)
    # return block

    proc_group = None
    rank = cfg.gpuid
    cfg.simcl.load = True
    cfg.simcl.rank = rank
    cfg.simcl.device = cfg.device
    cfg.simcl.denoising_prep = True
    simcl = simcl_load_model(cfg.simcl,rank,proc_group)
    for name,param in simcl.named_parameters():
        param = param.requires_grad_(False)
    simcl_enc = ClBlockEnc(simcl,cfg.dynamic.frame_size,cfg.d_model_attn)
    return simcl_enc


