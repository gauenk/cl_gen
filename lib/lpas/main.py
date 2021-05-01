"""
Aligned Burst Patch Search

"""

# -- python imports --
import sys,os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from easydict import EasyDict as edict

# -- pytorch import --
import torch
from torch import nn
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

# -- project code --
import settings
from pyutils.timer import Timer
from datasets import load_dataset
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr,count_parameters

# -- [this folder] project code --
from .utils import get_ref_block_index,get_block_arangements
from .config import get_cfg,get_args
from .explore_fast_unet import fast_unet
from .eval_score import eval_score
from .explore_cog import explore_cog

def get_main_config(rank=0,Sgrid=[1],Ngrid=[3],nNgrid=1,Ggrid=[25.],nGgrid=1,ngpus=3,idx=0):
    
    args = get_args()
    args.name = "default"
    cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.use_apex = False
    gpuid = rank % ngpus # set gpuid
    gpuid = 1
    cfg.gpuid = gpuid
    cfg.device = f"cuda:{gpuid}"

    grid_idx = idx*(1*ngpus)+rank
    B_grid_idx = (grid_idx % 2)
    N_grid_idx = ( grid_idx // 2 ) % nNgrid
    G_grid_idx = grid_idx // (nNgrid * 2) % nGgrid
    S_grid_idx = grid_idx // (nGgrid * nNgrid * 2) 

    # -- random seeding --
    cfg.set_worker_seed = False
    if cfg.set_worker_seed:
        cfg.dynamic.reset_seed = True
        torch.manual_seed(123)

    # -- force blind --
    B_grid_idx = 0

    # -- config settings --
    cfg.use_collate = True
    # cfg.dataset.download = False
    # cfg.cls = cfg
    cfg.S = Sgrid[S_grid_idx]
    # cfg.dataset.name = "cifar10"
    # cfg.dataset.name = "cbsd68"
    cfg.dataset.name = "voc"
    # cfg.dataset.name = "sun2009"
    # cfg.dataset.name = "eccv2020"
    # cfg.dataset.name = "rebel2021"
    cfg.supervised = False
    cfg.n2n = True
    cfg.blind = (B_grid_idx == 0)
    cfg.blind = ~cfg.supervised
    cfg.N = Ngrid[N_grid_idx]
    cfg.N = 5
    cfg.nframes = cfg.N
    cfg.sim_only_middle = True
    cfg.use_kindex_lmdb = True
    cfg.num_workers = 0
    cfg.frame_size = 216
    cfg.zero_mean_images = True

    # -- abp search params --
    cfg.patchsize = 15
    cfg.nblocks = 5
    cfg.nh_size = cfg.nblocks

    # -- kpn params --
    cfg.kpn_filter_onehot = False
    cfg.kpn_1f_frame_size = 2
    cfg.kpn_frame_size = 5

    cfg.kpn_cascade = True
    cfg.kpn_cascade_output = False
    cfg.kpn_num_frames = 3
    cfg.kpn_cascade_num = 3

    cfg.kpn_1f_cascade = False
    cfg.kpn_1f_cascade_output = False
    cfg.kpn_1f_cascade_num = 1

    cfg.burst_use_alignment = False
    cfg.burst_use_unet = True
    cfg.burst_use_unet_only = True
    cfg.kpn_burst_alpha = 0.998

    # -- noise-2-similar parameters --
    cfg.sim_shuffleK = True
    cfg.sim_method = "l2"
    cfg.sim_K = 8
    cfg.sim_patchsize = 7

    # -- byol parameters --
    cfg.byol_patchsize = 7
    cfg.byol_nh_size = 9
    cfg.byol_in_ftr_size = 3*cfg.byol_nh_size**2
    cfg.byol_out_ftr_size = 5 # 3*cfg.byol_patchsize**2
    cfg.byol_st_cat = 'v1'
    cfg.byol_num_test_samples = 1    
    cfg.byol_num_train_rand_crop = 1
    # cfg.byol_backbone_name = "attn"
    cfg.byol_backbone_name = "unet"
    byol_str = f"[BYOL]: {cfg.byol_backbone_name} {cfg.byol_patchsize} {cfg.byol_nh_size} {cfg.byol_in_ftr_size} {cfg.byol_out_ftr_size}"
    print(byol_str)


    # -- dataset params --
    cfg.dataset.triplet_loader = False
    cfg.dataset.dict_loader = True
    
    # -- gaussian noise --
    # cfg.noise_type = 'g'
    # cfg.noise_params.ntype = cfg.noise_type
    # cfg.noise_params['g']['stddev'] = Ggrid[G_grid_idx]
    # noise_level = Ggrid[G_grid_idx] # don't worry about
    # noise_level_str = f"{int(noise_level)}"

    # -- low-light noise --
    # noise_type = "qis"
    # cfg.noise_type = noise_type
    # cfg.noise_params['qis']['alpha'] = 4.0
    # cfg.noise_params['qis']['readout'] = 0.0
    # cfg.noise_params['qis']['nbits'] = 3
    # cfg.noise_params.ntype = cfg.noise_type

    # -- no noise for burst --
    noise_type = "none"
    cfg.noise_type = noise_type
    cfg.noise_params.ntype = cfg.noise_type
    cfg.noise_params.none = edict()

    # cfg.N = 30
    cfg.dynamic.frames = cfg.N
    cfg.batch_size = 2
    cfg.init_lr = 5e-5 # used to be 5e-4, 03/27/2020
    cfg.unet_channels = 3
    cfg.input_N = cfg.N-1
    cfg.epochs = 100
    cfg.color_cat = True
    cfg.log_interval = 300 #int(int(50000 / cfg.batch_size) / 500)
    cfg.save_interval = 2
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 1
    cfg.dynamic.random_eraser = False
    cfg.dynamic.frame_size = cfg.frame_size
    cfg.dynamic.total_pixels = cfg.dynamic.ppf*(cfg.N-1)
    
    # -- asdf --
    cfg.solver = edict()
    cfg.solver.max_iter = cfg.epochs* ( (721*16)/cfg.batch_size )
    cfg.solver.ramp_up_fraction = 0.1
    cfg.solver.ramp_down_fraction = 0.3

    # -- load previous experiment --
    cfg.load_epoch = 0
    cfg.load = cfg.load_epoch > 0
    cfg.restart_after_load = False
    return cfg

def run_me(rank=0,Sgrid=[1],Ngrid=[3],nNgrid=1,Ggrid=[25.],nGgrid=1,ngpus=3,idx=0):
    
    cfg = get_main_config()

    # -- noise info --
    noise_type = cfg.noise_params.ntype
    noise_params = cfg.noise_params['qis']
    noise_level = noise_params['readout']
    noise_level_str = f"{int(noise_params['alpha']),int(noise_params['readout']),int(noise_params['nbits'])}"

    # -- experiment info --
    name = "abps_v1p0"
    ds_name = cfg.dataset.name.lower()
    sup_str = "sup" if cfg.supervised else "unsup"
    bs_str = "b{}".format(cfg.batch_size)
    align_str = "yesAlignNet" if cfg.burst_use_alignment else "noAlignNet"
    unet_str = "yesUnet" if cfg.burst_use_unet else "noUnet"
    if cfg.burst_use_unet_only: unet_str += "Only"
    kpn_cascade_str = "cascade{}".format(cfg.kpn_cascade_num) if cfg.kpn_cascade else "noCascade"
    kpnba_str = "kpnBurstAlpha{}".format(int(cfg.kpn_burst_alpha*1000))
    frame_str = "n{}".format(cfg.N)
    framesize_str = "f{}".format(cfg.dynamic.frame_size)
    filtersize_str = "filterSized{}".format(cfg.kpn_frame_size)
    misc = "noKL"
    cfg.exp_name = f"{sup_str}_{name}_{ds_name}_{kpn_cascade_str}_{bs_str}_{frame_str}_{framesize_str}_{filtersize_str}_{align_str}_{unet_str}_{kpnba_str}_{misc}"
    print(f"Experiment name: {cfg.exp_name}")
    desc_fmt = (frame_str,kpn_cascade_str,framesize_str,filtersize_str,cfg.init_lr,align_str)
    cfg.desc = "Desc: unsup, frames {}, cascade {}, framesize {}, filter size {}, lr {}, {}, kl loss, anneal mse".format(*desc_fmt)
    print(f"Description: [{cfg.desc}]")
    noise_level = cfg.noise_params['g']['stddev']

    # -- attn params --
    cfg.patch_sizes = [128,128]
    cfg.d_model_attn = 3

    cfg.input_noise = False
    cfg.input_noise_middle_only = False
    cfg.input_with_middle_frame = True

    cfg.middle_frame_random_erase = False
    cfg.input_noise_level = noise_level/255.
    if (cfg.blind == 0): # e.g. supervised is true
        cfg.input_with_middle_frame = True
    if cfg.input_with_middle_frame:
        cfg.input_N = cfg.N

    blind = "blind" if cfg.blind else "nonblind"
    gpuid = cfg.gpuid
    print(blind,cfg.N,noise_level,gpuid)

    # if blind == "nonblind": return 
    dynamic_str = "dynamic_input_noise" if cfg.input_noise else "dynamic"
    if cfg.input_noise_middle_only: dynamic_str += "_mo"
    if cfg.input_with_middle_frame: dynamic_str += "_wmf"
    postfix = Path(f"./modelBurst/{cfg.exp_name}/{dynamic_str}/{cfg.dynamic.frame_size}_{cfg.dynamic.ppf}_{cfg.dynamic.total_pixels}/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    print(postfix,cfg.dynamic.total_pixels)
    cfg.model_path = cfg.model_path / postfix
    cfg.optim_path = cfg.optim_path / postfix
    if not cfg.model_path.exists(): cfg.model_path.mkdir(parents=True)
    if not cfg.optim_path.exists(): cfg.optim_path.mkdir(parents=True)
    
    checkpoint = cfg.model_path / Path("checkpoint_{}.tar".format(cfg.epochs))
    # if checkpoint.exists(): return

    torch.cuda.set_device(gpuid)


    data,loader = load_dataset(cfg,'denoising')
    explore_cog(cfg,data,overwrite=False)
    # data,loader = load_dataset(cfg,'dynamic')
    # fast_unet(cfg,data,overwrite=False)
    # eval_score(cfg,data,overwrite=True)

