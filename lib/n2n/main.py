
# -- python imports --
import os
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
from .model_io import load_model,load_model_fp
from .optim_io import load_optimizer
from .sched_io import load_scheduler
from .learn import train_loop,test_loop
from .learn_n2n import train_loop_n2n,test_loop_n2n
from .ot_loss import *

def run_me(rank=0,Sgrid=[1],Ngrid=[3],nNgrid=1,Ggrid=[75.],nGgrid=1,ngpus=3,idx=0):
    
    args = get_args()
    args.name = "default"
    cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.use_apex = False
    cfg.global_steps = 0
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
    cfg.dataset.name = "voc"
    # cfg.dataset.name = "rebel2021"
    cfg.blind = (B_grid_idx == 0)
    cfg.blind = False
    cfg.N = Ngrid[N_grid_idx]
    cfg.N = 2
    cfg.use_anscombe = True

    # -- noise 2 simulate parameters --
    cfg.sim_shuffleK = True
    cfg.sim_method = "l2"
    cfg.sim_K = 1
    cfg.sim_patchsize = 9
    # cfg.N = 30
    cfg.num_workers = 2
    cfg.dynamic.frames = cfg.N

    # -- gaussian noise --
    # noise_type = "g"
    # cfg.noise_type = noise_type
    # cfg.noise_params['g']['stddev'] = Ggrid[G_grid_idx]
    # cfg.noise_params.ntype = cfg.noise_type
    # noise_params = cfg.noise_params['g']
    # noise_level = Ggrid[G_grid_idx]
    # noise_level_str = f"{int(noise_level)}"

    # -- heteroskedastic gaussian noise --
    # noise_type = "hg"
    # cfg.noise_type = noise_type
    # cfg.noise_params['hg']['read'] = Ggrid[G_grid_idx]
    # cfg.noise_params['hg']['shot'] = 25.
    # noise_params = cfg.noise_params['hg']
    # cfg.noise_params.ntype = cfg.noise_type
    # noise_level = Ggrid[G_grid_idx]
    # noise_level_str = f"{int(noise_params['read']),int(noise_params['shot'])}"

    # -- low-light noise --
    noise_type = "qis"
    cfg.noise_type = noise_type
    cfg.noise_params['qis']['alpha'] = 4.0
    cfg.noise_params['qis']['readout'] = 0.0
    cfg.noise_params['qis']['nbits'] = 3
    noise_params = cfg.noise_params['qis']
    cfg.noise_params.ntype = cfg.noise_type
    noise_level = noise_params['readout']
    noise_level_str = f"{int(noise_params['alpha']),int(noise_params['readout']),int(noise_params['nbits'])}"

    # -- batch info --
    cfg.batch_size = 4
    cfg.init_lr = 1e-4
    cfg.unet_channels = 3
    cfg.input_N = cfg.N-1
    cfg.epochs = 50
    cfg.color_cat = True
    cfg.log_interval = int(int(50000 / cfg.batch_size) / 100)
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 0
    cfg.dynamic.random_eraser = False
    cfg.dynamic.frame_size = 128
    cfg.dynamic.total_pixels = cfg.dynamic.ppf * cfg.N
    # cfg.dynamic.total_pixels = 6
    cfg.load = False
    print("This is to benchmark a new distribution class. We want to find the case when mL2 fails.")

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
    postfix = Path(f"./{dynamic_str}/{cfg.dynamic.frame_size}_{cfg.dynamic.ppf}_{cfg.dynamic.total_pixels}/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    print(postfix,cfg.dynamic.total_pixels)
    cfg.model_path = cfg.model_path / postfix
    cfg.optim_path = cfg.optim_path / postfix
    if not cfg.model_path.exists(): cfg.model_path.mkdir(parents=True)
    if not cfg.optim_path.exists(): cfg.optim_path.mkdir(parents=True)
    
    checkpoint = cfg.model_path / Path("checkpoint_{}.tar".format(cfg.epochs))
    # if checkpoint.exists(): return
    print(f"Sim Method: {cfg.sim_method} | Shuffle K {cfg.sim_shuffleK} | Sim K: {cfg.sim_K} | Patchsize: {cfg.sim_patchsize}")
    print(f"Ascombe Transform: {cfg.use_anscombe}")
    print("N: {} | Noise Level: {} | Noise Type: {}".format(cfg.N,noise_level_str,noise_type))
    print("PID: {}".format(os.getpid()))
    torch.cuda.set_device(gpuid)

    # load model
    torch.manual_seed(cfg.seed)
    model = load_model(cfg)
    optimizer = load_optimizer(cfg,model)
    scheduler = load_scheduler(cfg,model,optimizer)
    nparams = count_parameters(model)
    print("Number of Trainable Parameters: {}".format(nparams))

    # load data
    # data,loader = load_dataset(cfg,'denoising')
    data,loader = load_dataset(cfg,'dynamic')
    # data,loader = simulate_noisy_dataset(data,loaders,M,N)

    # load criterion
    criterion = nn.BCELoss()

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
        
    # run_ot_v_displacement(cfg,criterion,loader.te)
    for epoch in range(cfg.current_epoch,cfg.epochs):

        losses = train_loop_n2n(cfg,model,optimizer,criterion,loader.tr,epoch)
        ave_psnr = test_loop_n2n(cfg,model,criterion,loader.te,epoch)
        # losses = train_loop(cfg,model,optimizer,criterion,loader.tr,epoch)
        # ave_psnr = test_loop(cfg,model,criterion,loader.te,epoch)
        te_ave_psnr[epoch] = ave_psnr
        cfg.current_epoch += 1

    epochs,psnr = zip(*te_ave_psnr.items())
    best_index = np.argmax(psnr)
    best_epoch,best_psnr = epochs[best_index],psnr[best_index]
    print(f"Best Epoch {best_epoch} | Best PSNR {best_psnr} | N: {cfg.N} | Blind: {blind}")
    
    root = Path(f"{settings.ROOT_PATH}/output/n2n/{postfix}/")
    # if cfg.blind: root = root / Path(f"./blind/")
    # else: root = root / Path(f"./nonblind/")
    fn = Path(f"results.csv")

    if not root.exists(): root.mkdir(parents=True)
    path = root / fn
    with open(path,'w') as f:
        f.write("{:d},{:d},{:2.10e},{:d}\n".format(cfg.N,best_epoch,best_psnr,nparams))
    
    save_model(cfg, model, optimizer)

def run_me_Ngrid():
    ngpus = 3
    nprocs_per_gpu = 1
    nprocs = ngpus * nprocs_per_gpu 
    Sgrid = [50000]
    # Ggrid = [10,25,50,100,150,200]
    Ggrid = [25]
    Ngrid = [3,10,100]
    nNgrid = len(Ngrid)
    nGgrid = len(Ggrid)
    te_losses = dict.fromkeys(Ngrid)
    num_of_grids = 2 * len(Sgrid) * len(Ggrid) * len(Ngrid) // nprocs
    for idx in range(num_of_grids):
        # for gpuid in range(ngpus):
        #     run_me(gpuid,Sgrid,Ngrid,nNgrid,Ggrid,nGgrid,ngpus,idx)
        r = mp.spawn(run_me, nprocs=nprocs, args=(Sgrid,Ngrid,nNgrid,Ggrid,nGgrid,ngpus,idx))
    # idx = num_of_grids
    # remainder = num_of_grids % nprocs
    # r = mp.spawn(run_me, nprocs=remainder, args=(Sgrid,Ngrid,nNgrid,Ggrid,nGgrid,ngpus,idx))

"""


cifar10
[train_offset] noisy -> clean gives psnr of 35 on cifar10 fixed gaussian noise std = 10 @ epoch?
[train_offset] noisy -> clean gives psnr of 29.9 - 30.2 on cifar10 msg (0,50) @ epoch 45 - 55
[train_n2n] standard stuff [psnr: 6] @ epochs 10 - 145


-- scheme 1 --
noisy_imgs -> clean_img
noisy_imgs -> noise_params
noise_instance ~ clean_img
new_noisy_img = noise_instance(clean_img)
mse(other_noisy_img,new_noisy_img)

-- scheme 2 --
noisy_imgs -> clean_img
mse(other_noisy_img,clean_img)

train_offset with 10 frames: (with no dropout)

n_input | n_output | psnr @ 30ish  | psnr @1150
  9     |   1      |   33.66       |  34.08
  5     |   5      |   32.31       |  32.55
  1     |   9      |   29.71       |  30.00


 N  |  PSNR 
 2  |  30.06
 3  |  31.85
 30 |  36.95



train_offset with 10 frames: (with dropout)

n_input | n_output | psnr
  9     |   1      | 
  5     |   5      | 
  1     |   9      | 





"""
