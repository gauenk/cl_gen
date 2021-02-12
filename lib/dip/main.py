"""
Deep Image Prior
"""



# -- python imports --
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

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
from learning.utils import save_model

# -- [this folder] project code --
from .config import get_cfg,get_args
from .model_io import load_model_fp,load_model_skip,load_model_kpn,load_model_attn
from .optim_io import load_optimizer
from .sched_io import load_scheduler
from .learn import train_loop,test_loop
from .learn_attn import test_loop as test_loop_attn
from .learn_burst import test_loop as test_loop_burst
from .learn_kpn import test_loop as test_loop_kpn
from .loop import dip_loop

torch.manual_seed(0)

def run_me(rank=0,Sgrid=[1],Ngrid=[8],nNgrid=1,Ggrid=[25.],nGgrid=1,ngpus=3,idx=0):    
    args = get_args()
    args.name = "default"
    cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.use_apex = False
    gpuid = rank % ngpus # set gpuid
    gpuid = 0
    cfg.gpuid = gpuid
    cfg.device = f"cuda:{gpuid}"

    grid_idx = idx*(1*ngpus)+rank
    B_grid_idx = (grid_idx % 2)
    N_grid_idx = ( grid_idx // 2 ) % nNgrid
    G_grid_idx = grid_idx // (nNgrid * 2) % nGgrid
    S_grid_idx = grid_idx // (nGgrid * nNgrid * 2) 

    # -- force blind --
    B_grid_idx = 0

    # -- attn parameters --
    cfg.patch_sizes = [32,32]
    cfg.d_model_attn = 512 # 512

    # -- config settings --
    cfg.use_collate = True
    # cfg.dataset.download = False
    # cfg.cls = cfg
    cfg.S = Sgrid[S_grid_idx]
    # cfg.dataset.name = "cifar10"
    cfg.dataset.name = "voc"
    cfg.blind = (B_grid_idx == 0)
    cfg.N = Ngrid[N_grid_idx]
    cfg.N = 3
    cfg.dynamic.frames = cfg.N
    cfg.noise_type = 'g'
    cfg.noise_params['g']['stddev'] = Ggrid[G_grid_idx]
    noise_level = Ggrid[G_grid_idx] # don't worry about
    cfg.batch_size = 1
    cfg.init_lr = 1e-3
    cfg.unet_channels = 3
    cfg.input_N = cfg.N-1
    cfg.epochs = 30
    cfg.color_cat = True
    cfg.log_interval = int(int(50000 / cfg.batch_size) / 100)
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 2
    cfg.dynamic.random_eraser = False
    cfg.dynamic.frame_size = 32
    # cfg.dynamic.total_pixels = cfg.dynamic.ppf * cfg.N
    cfg.dynamic.total_pixels = 2 * cfg.N
    cfg.load = False

    cfg.input_noise = False
    cfg.middle_frame_random_erase = False
    cfg.input_noise_level = noise_level/255.
    cfg.input_with_middle_frame = True
    if (cfg.blind == 0): # e.g. supervised is true
        cfg.input_with_middle_frame = True
    if cfg.input_with_middle_frame:
        cfg.input_N = cfg.N

    blind = "blind" if cfg.blind else "nonblind"
    print(grid_idx,blind,cfg.N,Ggrid[G_grid_idx],gpuid)

    # if blind == "nonblind": return 
    dynamic_str = "dynamic_input_noise" if cfg.input_noise else "dynamic"
    if cfg.input_with_middle_frame: dynamic_str += "_wmf"
    postfix = Path(f"./{dynamic_str}/{cfg.dynamic.frame_size}_{cfg.dynamic.ppf}_{cfg.dynamic.total_pixels}/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    print(postfix,cfg.dynamic.total_pixels)
    cfg.model_path = cfg.model_path / postfix
    cfg.optim_path = cfg.optim_path / postfix
    if not cfg.model_path.exists(): cfg.model_path.mkdir(parents=True)
    if not cfg.optim_path.exists(): cfg.optim_path.mkdir(parents=True)
    
    checkpoint = cfg.model_path / Path("checkpoint_{}.tar".format(cfg.epochs))
    # if checkpoint.exists(): return

    print("N: {} | Noise Level: {}".format(cfg.N,cfg.noise_params['g']['stddev']))

    torch.cuda.set_device(gpuid)

    # load model
    # model,criterion = load_model(cfg)
    # model,criterion = load_model_kpn(cfg)
    model,criterion = load_model_attn(cfg)
    optimizer = load_optimizer(cfg,model)
    scheduler = load_scheduler(cfg,model,optimizer)
    nparams = count_parameters(model)
    print("Number of Trainable Parameters: {}".format(nparams))

    # load data
    # data,loader = load_dataset(cfg,'denoising')
    data,loader = load_dataset(cfg,'dynamic')
    # data,loader = simulate_noisy_dataset(data,loaders,M,N)

    # load criterion
    # criterion = nn.BCELoss()

    if cfg.load:
        fp = cfg.model_path / Path("checkpoint_30.tar")
        model = load_model_fp(cfg,model,fp,0)

    cfg.current_epoch = 0
    te_ave_psnr = {}
    test_before = False
    if test_before:
        ave_psnr = test_loop(cfg,model,criterion,loader.te,-1)
        print("PSNR before training {:2.3e}".format(ave_psnr))
        return 
    if checkpoint.exists() and cfg.load:
        model = load_model_fp(cfg,model,checkpoint,gpuid)
        print("Loaded model.")
        cfg.current_epoch = cfg.epochs
        
    for epoch in range(cfg.current_epoch,cfg.epochs):

        # losses = train_loop(cfg,model,optimizer,criterion,loader.tr,epoch)
        # ave_psnr = test_loop(cfg,model,optimizer,criterion,loader.tr,epoch)
        # ave_psnr = test_loop_burst(cfg,model,optimizer,criterion,loader.tr,epoch)
        # ave_psnr = test_loop_kpn(cfg,model,optimizer,criterion,loader.tr,epoch)
        # ave_psnr = test_loop_attn(cfg,model,optimizer,criterion,loader.tr,epoch)
        ave_psnr = dip_loop(cfg,loader.tr,epoch)
        te_ave_psnr[epoch] = ave_psnr
        cfg.current_epoch += 1

    epochs,psnr = zip(*te_ave_psnr.items())
    best_index = np.argmax(psnr)
    best_epoch,best_psnr = epochs[best_index],psnr[best_index]
    print(f"Best Epoch {best_epoch} | Best PSNR {best_psnr} | N: {cfg.N} | Blind: {blind}")
    
    root = Path(f"{settings.ROOT_PATH}/output/dip/{postfix}/")
    # if cfg.blind: root = root / Path(f"./blind/")
    # else: root = root / Path(f"./nonblind/")
    fn = Path(f"results.csv")

    if not root.exists(): root.mkdir(parents=True)
    path = root / fn
    with open(path,'w') as f:
        f.write("{:d},{:d},{:2.10e},{:d}\n".format(cfg.N,best_epoch,best_psnr,nparams))
    
    save_model(cfg, model, optimizer)
    
