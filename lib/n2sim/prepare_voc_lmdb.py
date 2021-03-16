
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

# -- project code --
import settings
from pyutils.timer import Timer
from datasets import load_dataset
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr,count_parameters
from datasets.prepare_pascal_voc import create_lmdb,create_lmdb_noisy_burst

# -- [this folder] project code --
from .config import get_cfg,get_args

def run_me(lmdb_type='all'):
    
    # -- init --
    rank=0
    Sgrid=[1]
    Ngrid=[3]
    nNgrid=1
    Ggrid=[75.]
    nGgrid=1
    ngpus=3
    idx=0

    # -- args + config --
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
    cfg.supervised = False
    cfg.blind = (B_grid_idx == 0)
    cfg.blind = ~cfg.supervised
    cfg.N = Ngrid[N_grid_idx]
    cfg.N = 6

    # -- kpn params --
    cfg.kpn_filter_onehot = False
    cfg.kpn_1f_frame_size = 2
    cfg.kpn_frame_size = 9
    cfg.kpn_cascade = False
    cfg.kpn_cascade_output = False
    cfg.kpn_cascade_num = 1
    cfg.burst_use_alignment = False
    cfg.burst_use_unet = False
    cfg.burst_use_unet_only = False
    

    # cfg.N = 30
    cfg.dynamic.frames = cfg.N
    cfg.noise_type = 'g'
    cfg.noise_params.ntype = cfg.noise_type
    cfg.noise_params['g']['stddev'] = Ggrid[G_grid_idx]
    noise_level = Ggrid[G_grid_idx] # don't worry about
    cfg.batch_size = 1
    cfg.init_lr = 5e-4
    cfg.unet_channels = 3
    cfg.input_N = cfg.N-1
    cfg.epochs = 300
    cfg.color_cat = True
    cfg.log_interval = 10 #int(int(50000 / cfg.batch_size) / 500)
    cfg.save_interval = 2
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 1
    cfg.dynamic.random_eraser = False
    cfg.dynamic.frame_size = 128
    cfg.dynamic.total_pixels = cfg.dynamic.ppf*cfg.N
    
    # -- asdf --
    cfg.solver = edict()
    cfg.solver.max_iter = cfg.epochs*500
    cfg.solver.ramp_up_fraction = 0.1
    cfg.solver.ramp_down_fraction = 0.3

    # -- load previous experiment --
    cfg.load_epoch = 0
    cfg.load = cfg.load_epoch > 0
    cfg.restart_after_load = True

    # -- experiment info --
    name = "n2sim_burst"
    sup_str = "sup" if cfg.supervised else "unsup"
    bs_str = "b{}".format(cfg.batch_size)
    align_str = "yesAlignNet" if cfg.burst_use_alignment else "noAlignNet"
    unet_str = "yesUnet" if cfg.burst_use_unet else "noUnet"
    kpn_cascade_str = "cascade{}".format(cfg.kpn_cascade_num) if cfg.kpn_cascade else "noCascade"
    frame_str = "n{}".format(cfg.N)
    framesize_str = "f{}".format(cfg.dynamic.frame_size)
    filtersize_str = "filterSized{}".format(cfg.kpn_frame_size)
    misc = "unet_mse_noKL"
    cfg.exp_name = f"{sup_str}_{name}_{kpn_cascade_str}_{bs_str}_{frame_str}_{framesize_str}_{filtersize_str}_{align_str}_{unet_str}_{misc}"
    print(f"Experiment name: {cfg.exp_name}")
    desc_fmt = (frame_str,kpn_cascade_str,framesize_str,filtersize_str,cfg.init_lr,align_str)
    cfg.desc = "Desc: unsup, frames {}, cascade {}, framesize {}, filter size {}, lr {}, {}, kl loss, anneal mse".format(*desc_fmt)
    print(f"Description: [{cfg.desc}]")

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
    print(grid_idx,blind,cfg.N,Ggrid[G_grid_idx],gpuid)

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

    print("N: {} | Noise Level: {}".format(cfg.N,cfg.noise_params['g']['stddev']))

    torch.cuda.set_device(gpuid)


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #   Load the Model, Data, Optim, Crit
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
    data,loader = load_dataset(cfg,'dynamic')
    ds_set = "train"
    # ds_set = "test"

    if lmdb_type == 'all':
        create_lmdb(cfg,data.tr,ds_set,epochs=1,maxNumFrames=cfg.N,numSim=10,patchsize=3)
    elif lmdb_type == 'burst':
        D = 100
        create_lmdb_noisy_burst(cfg,D,data.tr,ds_set,maxNumFrames=cfg.N,id_str='burst')
    elif lmdb_type == 'noise':
        raise NotImplemented(f"Not yet implemented lmdb type [{lmdb_type}]")
    else: raise ValueError(f"Uknown LMDB Type [{lmdb_type}]")
            
