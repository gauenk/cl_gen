
# -- python imports --
from pathlib import Path

# -- pytorch imports --
import torch

# -- project imports --
from layers.attn import TransformerNetwork32,TransformerNetwork32_noxform,TransformerNetwork32_v3
from layers.simcl import ClBlockEnc
from layers.unet import UNet_small,UNet_v2
from simcl.model_io import load_model as simcl_load_model

def load_model(cfg):
    if cfg.model_version == "v1":
        return load_model_v1(cfg)
    elif cfg.model_version == "v2":
        return load_model_v2(cfg)
    elif cfg.model_version == "v3":
        return load_model_v3(cfg)
    else:
        raise KeyError(f"Uknown Model Type [{cfg.model_version}]")

def load_model_v1(cfg):
    simclr = None #load_model_simclr(cfg)
    denoise_model = UNet_small(cfg.d_model_attn*cfg.input_N)
    # denoise_model = load_denoise_model_fp(cfg,denoise_model)

    input_patch,output_patch = cfg.patch_sizes
    # d_model = 2048 // (patch_height * patch_width)
    # d_model = 398336 // (patch_height * patch_width)
    d_model = cfg.d_model_attn    
    xformer_args = [simclr, d_model, cfg.dynamic.frame_size, input_patch,
                    output_patch, cfg.input_N, cfg.dataset.bw, denoise_model, cfg.batch_size]
    model = TransformerNetwork32(*xformer_args)
    model = model.to(cfg.device)
    return model

def load_model_v2(cfg):
    simclr = None
    denoise_model = UNet_small(3*cfg.input_N)
    input_patch,output_patch = cfg.patch_sizes
    d_model = cfg.d_model_attn    
    xformer_args = [simclr, d_model, cfg.dynamic.frame_size, input_patch,
                    output_patch, cfg.input_N, cfg.dataset.bw, denoise_model]
    model = TransformerNetwork32_noxform(*xformer_args)
    model = model.to(cfg.device)
    return model

def load_model_v3(cfg):
    simclr = None #load_model_simclr(cfg)
    denoise_model = UNet_small(cfg.d_model_attn*cfg.input_N)
    # denoise_model = load_denoise_model_fp(cfg,denoise_model)

    input_patch,output_patch = cfg.patch_sizes
    # d_model = 2048 // (patch_height * patch_width)
    # d_model = 398336 // (patch_height * patch_width)
    d_model = cfg.d_model_attn    
    xformer_args = [simclr, d_model, cfg.dynamic.frame_size, input_patch,
                    output_patch, cfg.input_N, cfg.dataset.bw, denoise_model]
    model = TransformerNetwork32_v3(*xformer_args)
    model = model.to(cfg.device)
    return model

def load_denoise_model_fp(cfg,denoise_model):
    model = load_model_v2(cfg)
    model_fp = "/home/gauenk/Documents/experiments/cl_gen/output/attn/voc/default_attn_16/model/v2/dynamic/16_0_0/-1/blind/3/25/checkpoint_451.tar"
    model = load_model_fp(cfg,model,model_fp,cfg.gpuid)
    dn = model.denoise_model
    nparam_l = dn.named_parameters()
    nparam_t = denoise_model.named_parameters()
    for ((name_l,param_l),(name_t,param_t)) in zip(nparam_l,nparam_t):
        if "conv1" in name_l: continue
        param_t.data = param_l.data
        param_t = param_t.requires_grad_(False)
    denoise_model = denoise_model.eval()
    return denoise_model

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


def load_model_fp(cfg,model,model_fp,rank):
    if cfg.use_ddp:
        map_location = {'cuda:%d' % 0, 'cuda:%d' % rank}
    else:
        map_location = 'cuda:%d' % rank
    print(f"Loading model filepath [{model_fp}]")
    state = torch.load(model_fp, map_location=map_location)
    model.load_state_dict(state)
    return model

