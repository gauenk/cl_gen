"""
Is my transformer doing the alignment I would expect?

"""

# -- python imports --
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import datetime

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

# -- project code --
import settings
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from datasets.transform import ScaleZeroMean

# -- [this folder] project code --
from .config import get_cfg,get_args
from .model_io import load_model,load_model_fp

def get_vis_alignment_exp_cfg():
    args = get_args()
    args.name = "default_attn_16"
    args.dataset = "voc"
    # args.num_workers = 0
    cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.use_apex = False
    gpuid = 1 # set gpuid
    cfg.gpuid = gpuid
    cfg.device = f"cuda:{gpuid}"
    cfg.use_collate = True
    cfg.dataset.download = False
    cfg.cls = cfg
    cfg.batch_size = 18

    cfg.blind = True
    cfg.N = 3
    cfg.dynamic.frames = cfg.N
    cfg.noise_type = 'g'
    cfg.noise_params['g']['stddev'] = 25
    noise_level = 25
    cfg.init_lr = 1e-3
    cfg.unet_channels = 3
    cfg.input_N = cfg.N-1
    cfg.epochs = 1000
    cfg.log_interval = int(int(50000 / cfg.batch_size) / 10) # NumOfBatches / Const = NumOfPrints
    cfg.save_epoch = 50

    # -- attn parameters --
    cfg.patch_sizes = [16,16]
    cfg.d_model_attn = 512

    cfg.dataset.name = "voc"
    cfg.dataset.bw = False
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 2
    cfg.dynamic.frame_size = 16
    cfg.dynamic.total_pixels = 6
    cfg.load = False

    # left; 5 total pix; 10 in 8 out 16 imsize; bs 64; gpuid 1; 26 PSNR @ 30 and 190 epochs
    # right; 20 total pix; 16 in 8 out 16 imsize; bs 12; gpuid 0; 21 PSNR @ 70 epochs
    # 16_2_20 ; 27.59 psnr 16 in? 16 out? 16 imsize
    # 10, 8 was also promising

    # -- model type --
    cfg.model_version = "v1"

    # -- input noise for learning --
    cfg.input_noise = False
    cfg.input_noise_level = 25./255
    cfg.input_with_middle_frame = False
    if cfg.input_with_middle_frame:
        cfg.input_N = cfg.N

    # -- learning --
    cfg.spoof_batch = False
    cfg.spoof_batch_size = 1

    torch.cuda.set_device(gpuid)

    # cfg.batch_size = 256
    cfg.init_lr = 5e-5
    return cfg


def run_vis_alignment_setup():


    # -- load config -- 
    cfg = get_vis_alignment_exp_cfg()

    # -- pick epoch for analysis --
    cfg.epochs = 100
    cfg.load = True

    # -- load model -- 
    model = load_model(cfg)

    # -- load data --
    data,loader = load_dataset(cfg,'dynamic')

    # -- path manipulation --
    blind = "blind" if cfg.blind else "nonblind"
    dynamic_str = "dynamic_input_noise" if cfg.input_noise else "dynamic"
    if cfg.input_with_middle_frame: dynamic_str += "_wmf"
    postfix = Path(f"./{cfg.model_version}/{dynamic_str}/{cfg.dynamic.frame_size}_{cfg.dynamic.ppf}_{cfg.dynamic.total_pixels}/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    print(f"model dir [{postfix}]")
    cfg.model_path = cfg.model_path / postfix
    cfg.optim_path = cfg.optim_path / postfix
    if not cfg.model_path.exists(): cfg.model_path.mkdir(parents=True)
    if not cfg.optim_path.exists(): cfg.optim_path.mkdir(parents=True)
    checkpoint = cfg.model_path / Path("checkpoint_{}.tar".format(cfg.epochs))
    print("N: {} | Noise Level: {}".format(cfg.N,cfg.noise_params['g']['stddev']))

    # -- load model from checkpoint --
    print(f"Loading from checkpoint: {checkpoint}")
    if cfg.load: model = load_model_fp(cfg,model,fp,cfg.gpuid)
    
    return cfg,model,data,loader

def run_vis_alignment():
    
    # -- init variables --
    cfg,model,data,loader = run_vis_alignment_setup()

    print(model)
