
# -- python imports --
import sys,os
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
from pyutils import np_log,rescale_noisy_image,mse_to_psnr,count_parameters
from learning.utils import save_model

# -- [this folder] project code --
from .config import get_cfg,get_args
from .model_io import load_model_kpn,load_model_fp,load_model
from .optim_io import load_optimizer_kpn as load_optimizer
from .sched_io import load_scheduler
from .learn_kpn import train_loop,test_loop
from .utils import init_record
from .ot_loss import run_test_xbatch,run_ot_v_displacement

def run_me(rank=0,Sgrid=[50000],Ngrid=[3],nNgrid=1,Ggrid=[1.],nGgrid=1,ngpus=3,idx=0):
# def run_me(rank=1,Ngrid=1,Ggrid=1,nNgrid=1,ngpus=3,idx=1):
    
    """
    PSNR 20 = (can equal) = AWGN @ 25
    PSNR 25 = (can equal) = AWGN @ 14
    PSNR 28 = (can equal) = AWGN @ 5
    """

    args = get_args()
    args.name = "default"
    cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.use_apex = False
    gpuid = 1
    cfg.gpuid = gpuid
    # gpuid = rank % ngpus # set gpuid
    cfg.device = f"cuda:{gpuid}"
    
    # -- experiment info --
    cfg.exp_name = "sup_n9_kpn-standard-filterSize15_f128_kpnLoss"
    cfg.desc = "Desc: sup kpn-standard-filterSized15, f128, kpnLoss"

    grid_idx = idx*(1*ngpus)+rank
    B_grid_idx = (grid_idx % 2)
    N_grid_idx = ( grid_idx // 2 ) % nNgrid
    G_grid_idx = grid_idx // (nNgrid * 2) % nGgrid
    S_grid_idx = grid_idx // (nGgrid * nNgrid * 2) 

    cfg.use_collate = True
    # cfg.dataset.download = False
    # cfg.cls = cfg
    cfg.S = Sgrid[S_grid_idx]
    # cfg.dataset.name = "cifar10"
    cfg.dataset.name = "voc"
    # cfg.blind = (B_grid_idx == 0)
    cfg.supervised = True
    cfg.blind = not cfg.supervised
    cfg.N = Ngrid[N_grid_idx]
    cfg.N = 6
    cfg.kpn_filter_onehot = True
    cfg.kpn_frame_size = 15
    cfg.dynamic.frames = cfg.N
    cfg.noise_type = 'g'
    cfg.noise_params['g']['stddev'] = Ggrid[G_grid_idx]
    # cfg.noise_type = 'll'
    # cfg.noise_params['ll']['alpha'] = 255*0.015
    # cfg.noise_params['ll']['read_noise'] = 0.25
    # cfg.recon_l1 = True
    noise_level = Ggrid[G_grid_idx]
    cfg.batch_size = 4
    cfg.init_lr = 1e-4
    cfg.unet_channels = 3
    cfg.input_N = cfg.N-1
    cfg.epochs = 100
    cfg.log_interval = 50 # int(int(50000 / cfg.batch_size) / 100)
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 2
    cfg.dynamic.frame_size = 128
    cfg.dynamic.total_pixels = 2*cfg.N
    cfg.load = False

    # -- input noise for learning --
    cfg.input_noise = False
    cfg.input_noise_middle_only = False
    cfg.input_with_middle_frame = True
    cfg.input_noise_level = noise_level/255
    if cfg.input_with_middle_frame:
        cfg.input_N = cfg.N

    blind = "blind" if cfg.blind else "nonblind"
    print(grid_idx,blind,cfg.N,Ggrid[G_grid_idx],gpuid,cfg.input_noise,cfg.input_with_middle_frame)

    # if blind == "nonblind": return 
    dynamic_str = "dynamic_input_noise" if cfg.input_noise else "dynamic"
    if cfg.input_noise_middle_only: dynamic_str += "_mo"
    if cfg.input_with_middle_frame: dynamic_str += "_wmf"
    postfix = Path(f"./{dynamic_str}/{cfg.dynamic.frame_size}_{cfg.dynamic.ppf}_{cfg.dynamic.total_pixels}/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    print(postfix)
    cfg.model_path = cfg.model_path / postfix
    cfg.optim_path = cfg.optim_path / postfix
    if not cfg.model_path.exists(): cfg.model_path.mkdir(parents=True)
    if not cfg.optim_path.exists(): cfg.optim_path.mkdir(parents=True)
    
    checkpoint = cfg.model_path / Path("checkpoint_{}.tar".format(cfg.epochs))
    # if checkpoint.exists(): return

    print("PID: {}".format(os.getpid()))
    print("N: {} | Noise Level: {}".format(cfg.N,cfg.noise_params['g']['stddev']))

    torch.cuda.set_device(gpuid)

    # -- load model --
    model,criterion = load_model_kpn(cfg)
    optimizer = load_optimizer(cfg,model)
    scheduler = load_scheduler(cfg,model,optimizer)
    nparams = count_parameters(model)
    print("Number of Trainable Parameters: {}".format(nparams))

    # load data
    # data,loader = load_dataset(cfg,'denoising')
    data,loader = load_dataset(cfg,'dynamic')
    # data,loader = simulate_noisy_dataset(data,loaders,M,N)


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
        
    cfg.global_step = 0
    use_record = False
    record = init_record()
    # run_test_xbatch(cfg,criterion,loader.tr)
    # run_ot_v_displacement(cfg,criterion,loader.tr)
    # exit()

    for epoch in range(cfg.current_epoch,cfg.epochs):

        print(cfg.desc)
        sys.stdout.flush()

        losses,epoch_record = train_loop(cfg,model,optimizer,criterion,loader.tr,epoch)

        if use_record:
            record = record.append(epoch_record)
            write_record_file(cfg.current_epoch,postfix,record)

        ave_psnr = test_loop(cfg,model,criterion,loader.te,epoch)
        te_ave_psnr[epoch] = ave_psnr
        cfg.current_epoch += 1


    epochs,psnr = zip(*te_ave_psnr.items())
    best_index = np.argmax(psnr)
    best_epoch,best_psnr = epochs[best_index],psnr[best_index]
    
    root = Path(f"{settings.ROOT_PATH}/output/n2n-kpn/{postfix}/")
    # if cfg.blind: root = root / Path(f"./blind/")
    # else: root = root / Path(f"./nonblind/")
    fn = Path(f"results.csv")

    if not root.exists(): root.mkdir(parents=True)
    path = root / fn
    with open(path,'w') as f:
        f.write("{:d},{:d},{:2.10e},{:d}\n".format(cfg.N,best_epoch,best_psnr,nparams))
    
    save_model(cfg, model, optimizer)

def run_me_grid():
    ngpus = 2
    nprocs_per_gpu = 1
    nprocs = ngpus * nprocs_per_gpu 
    Sgrid = [50000]
    # Ggrid = [10,25,50,100,150,200]
    Ggrid = [25]
    Ngrid = [10,5,20]
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

def write_record_file(current_epoch,postfix,record):
    root = Path(f"{settings.ROOT_PATH}/output/n2n-kpn/{postfix}/")
    if not root.exists(): root.mkdir(parents=True)
    path = root / f"record_unsupOT-xbatch_plus_unsupMSE_{current_epoch}.csv"
    print(f"Writing record_losses to {path}")
    record.to_csv(path)


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
