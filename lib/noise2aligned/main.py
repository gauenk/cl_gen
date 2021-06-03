
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
from pyutils import np_log,rescale_noisy_image,mse_to_psnr,count_parameters

# -- [this folder] project code --
from .config import get_cfg,get_args
from .model_io import load_unet_model,load_model_fp,load_model_kpn,load_burst_n2n_model,load_burst_kpn_model,save_burst_model
from .optim_io import load_optimizer,load_optim_fp
from .sched_io import load_scheduler,make_lr_scheduler
from .learn import train_loop,test_loop


def set_seed(cfg):
    if cfg.use_seed:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.set_deterministic(True)
    
def get_main_config(rank=0,Sgrid=[1],Ngrid=[3],nNgrid=1,Ggrid=[25.],nGgrid=1,ngpus=3,idx=0):
    
    args = get_args()
    args.name = "default"
    cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.use_apex = False
    gpuid = rank % ngpus # set gpuid
    gpuid = 0
    cfg.gpuid = gpuid
    cfg.device = f"cuda:{gpuid}"
    cfg.seed = 123
    cfg.use_seed = True
    # cfg.lpas_method = "exhaustive"
    # cfg.lpas_method = "simple"
    # cfg.lpas_method = "spoof"
    cfg.lpas_method = "split"
        
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
    # cfg.dataset.name = "eccv2020"
    # cfg.dataset.name = "rebel2021"
    cfg.supervised = False
    cfg.n2n = False
    cfg.abps = True
    cfg.abps_inputs = False
    cfg.blind = (B_grid_idx == 0)
    cfg.blind = ~cfg.supervised
    cfg.N = Ngrid[N_grid_idx]
    cfg.N = 3
    cfg.nframes = cfg.N
    cfg.sim_only_middle = True
    cfg.use_kindex_lmdb = True
    cfg.num_workers = 4

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
    cfg.burst_use_unet = False
    cfg.burst_use_unet_only = False
    cfg.kpn_burst_alpha = 0.998

    # -- abp search parameters --
    cfg.patchsize = 15
    cfg.nblocks = 5
    cfg.nh_size = cfg.nblocks

    # -- noise-2-similar parameters --
    cfg.sim_shuffleK = True
    cfg.sim_method = "l2"
    cfg.sim_K = 8
    cfg.sim_patchsize = 9

    # -- dataset params --
    cfg.dataset.triplet_loader = False
    cfg.dataset.dict_loader = True
    
    # -- gaussian noise --
    # cfg.noise_type = 'g'
    # noise_level = 50.0
    # cfg.noise_params.ntype = cfg.noise_type
    # cfg.noise_params['g']['stddev'] = noise_level
    # #noise_level = Ggrid[G_grid_idx] # don't worry about
    # noise_level_str = f"{int(noise_level)}"

    # -- low-light noise --
    noise_type = "qis"
    cfg.noise_type = noise_type
    cfg.noise_params['qis']['alpha'] = 4.0
    cfg.noise_params['qis']['readout'] = 0.0
    cfg.noise_params['qis']['nbits'] = 3
    cfg.noise_params.ntype = cfg.noise_type

    # cfg.N = 30
    # if cfg.abps: cfg.dynamic.frames = cfg.N + 1
    # else: cfg.dynamic.frames = cfg.N
    cfg.dynamic.frames = cfg.N + 1
    cfg.batch_size = 4
    cfg.init_lr = 1e-4 # used to be 5e-4, 03/27/2020
    cfg.unet_channels = 3
    cfg.input_N = cfg.N-1
    cfg.epochs = 100
    cfg.color_cat = True
    cfg.log_interval = 25 #int(int(50000 / cfg.batch_size) / 500)
    cfg.save_interval = 2
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 1
    cfg.dynamic.random_eraser = False
    cfg.dynamic.total_pixels = cfg.dynamic.ppf*(cfg.dynamic.frames-1)
    cfg.dataset.load_residual = True
    cfg.dataset.reset_seed = False

    # -- frame size --
    # if cfg.abps: 
    # else: M = 0
    M = (1+cfg.dynamic.ppf)*cfg.nframes
    cfg.frame_size = 128
    cfg.dynamic.frame_size = cfg.frame_size + 2*M
    
    # -- asdf --
    cfg.solver = edict()
    cfg.solver.max_iter = cfg.epochs* ( (721*16)/cfg.batch_size )
    cfg.solver.ramp_up_fraction = 0.1
    cfg.solver.ramp_down_fraction = 0.3

    # -- load previous experiment --
    cfg.load_epoch = 0
    cfg.load = cfg.load_epoch > 0
    cfg.restart_after_load = False

    # -- spoof optical flow test --
    cfg.optical_flow_acc = 1.0
    print("Optical Flow Acc: %2.2f" % (100. * cfg.optical_flow_acc))
    return cfg

def run_me(rank=0,Sgrid=[1],Ngrid=[3],nNgrid=1,Ggrid=[25.],nGgrid=1,ngpus=3,idx=0):
    
    cfg = get_main_config()

    # -- noise info --
    noise_type = cfg.noise_params.ntype
    noise_params = cfg.noise_params['g']
    noise_level = noise_params['stddev']
    noise_level_str = f"{int(noise_params['stddev'])}"

    # noise_params = cfg.noise_params['qis']
    # noise_level = noise_params['readout']
    # noise_level_str = f"{int(noise_params['alpha']),int(noise_params['readout']),int(noise_params['nbits'])}"

    # -- experiment info --
    name = "noise2aligned_burstv2_testingAlignedAbps"
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

    print(f"Supervised: {cfg.supervised} | Noise2Noise: {cfg.n2n} | APBS: {cfg.abps} | ABPS-Inputs: {cfg.abps_inputs}")
    print(f"Sim Method: {cfg.sim_method} | Shuffle K {cfg.sim_shuffleK} | Sim K: {cfg.sim_K} | Patchsize: {cfg.sim_patchsize}")
    print("N: {} | Noise Level: {} | Noise Type: {}".format(cfg.N,noise_level_str,noise_type))

    torch.cuda.set_device(gpuid)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #   init summary writer
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    log_base = Path(f"runs/{name}")
    if not log_base.exists(): log_base.mkdir(parents=True)
    log_dir = log_base / Path(f"{cfg.exp_name}")
    writer = SummaryWriter(log_dir=str(log_dir))

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #   Load the Model, Data, Optim, Crit
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
    # load model
    # model = load_unet_model(cfg)
    # model,criterion = load_burst_n2n_model(cfg)
    if cfg.use_seed: set_seed(cfg)
    model,noise_critic,criterion = load_burst_kpn_model(cfg)
    # model,criterion = load_model_kpn(cfg)
    # optimizer = load_optimizer(cfg,model)
    # scheduler = load_scheduler(cfg,model,optimizer)
    # scheduler = make_lr_scheduler(cfg,model.unet_info.optim)
    nparams = count_parameters(model.denoiser_info.model)
    print("Number of Trainable Parameters: {}".format(nparams))
    print("GPUID: {}".format(gpuid))
    print("PID: {}".format(os.getpid()))

    # load data
    # data,loader = load_dataset(cfg,'denoising')
    # data,loader = load_dataset(cfg,'default')
    data,loader = load_dataset(cfg,'dynamic')
    # data,loader = load_dataset(cfg,'dynamic-lmdb-all')
    # data,loader = load_dataset(cfg,'dynamic-lmdb-burst')
    # data,loader = load_dataset(cfg,'default')
    # data,loader = simulate_noisy_dataset(data,loaders,M,N)

    # load criterion
    # criterion = nn.BCELoss()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #    Load the Model from Memory
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


    if cfg.load:
        name = "denoiser"
        fp = "/home/gauenk/Documents/experiments/cl_gen/output/noise2aligned/cifar10/default/model/modelBurst/unsup_noise2aligned_burstv2_voc_noCascade_b20_n8_f128_filterSized9_noAlignNet_noUnet_unet_mse_noKL/dynamic_wmf/128_1_8/1/blind/8/25.0/denoiser/checkpoint_83.tar"
        # fp = cfg.model_path / Path("{}/checkpoint_{}.tar".format(name,cfg.load_epoch))
        # fp = Path("/home/gauenk/Documents/experiments/cl_gen/output/n2n_wl/cifar10/default/model/modelBurst/unsup_burst_noCascade_b4_n10_f128_filterSized12_kpn_klLoss_annealMSE_klPRes/dynamic_wmf/128_1_10/1/blind/10/25.0/denoiser/checkpoint_{}.tar".format(cfg.load_epoch))
        model.denoiser_info.model = load_model_fp(cfg,model.denoiser_info.model,fp,cfg.gpuid)
        fp = "/home/gauenk/Documents/experiments/cl_gen/output/noise2aligned/cifar10/default/optim/modelBurst/unsup_noise2aligned_burstv2_voc_noCascade_b20_n8_f128_filterSized9_noAlignNet_noUnet_unet_mse_noKL/dynamic_wmf/128_1_8/1/blind/8/25.0/denoiser/checkpoint_83.tar"
        # model.denoiser_info.optim = load_optim_fp(cfg,model.denoiser_info.optim,fp,cfg.gpuid)
        # name = "critic"
        # fp = cfg.model_path / Path("{}/checkpoint_{}.tar".format(name,cfg.load_epoch))
        # noise_critic.disc = load_model_fp(cfg,noise_critic.disc,fp,cfg.gpuid)
        if cfg.restart_after_load:
            cfg.current_epoch = 0
            cfg.global_step = 0
        else:
            cfg.current_epoch = cfg.load_epoch+1
            cfg.global_step = cfg.load_epoch * len(data.tr)
            ce,gs = cfg.current_epoch,cfg.global_step
            print(f"Starting Training from epoch [{ce}] and global step [{gs}]")
    else:
        cfg.current_epoch = 0
    scheduler = make_lr_scheduler(cfg,model.denoiser_info.optim,cfg.global_step)


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       Pre train-loop setup
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    te_ave_psnr = {}
    test_before = False
    if test_before:
        ave_psnr,record_test = test_loop(cfg,model,loader.te,-1)
        print("PSNR before training {:2.3e}".format(ave_psnr))
        return 
    if checkpoint.exists() and cfg.load:
        model = load_model_fp(cfg,model,checkpoint,cfg.gpuid)
        print("Loaded model.")
        cfg.current_epoch = cfg.epochs+1
        cfg.global_step = len(train_data) * cfg.epochs
        
    record_losses = pd.DataFrame({'kpn':[],'ot':[],'psnr':[],'psnr_std':[]})
    use_record = False
    loss_type = "sup_r_ot"

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       Training Loop
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    for epoch in range(cfg.current_epoch,cfg.epochs):
        lr = model.denoiser_info.optim.param_groups[0]["lr"]
        print(cfg.desc)
        print("Learning Rate: %2.2e"% (lr))
        sys.stdout.flush()

        losses,record_losses = train_loop(cfg,model,scheduler,loader.tr,epoch,record_losses,writer)
        if use_record:
            write_record_losses_file(cfg.current_epoch,postfix,loss_type,record_losses)

        cfg.current_epoch += 1
        if epoch % cfg.save_interval == 0 and epoch > 0:
            save_burst_model(cfg,"align",model.align_info.model,model.align_info.optim)
            save_burst_model(cfg,"denoiser",model.denoiser_info.model,model.denoiser_info.optim)
            save_burst_model(cfg,"critic",noise_critic.disc,noise_critic.optim)

        ave_psnr,record_test = test_loop(cfg,model,loader.te,epoch)
        if use_record:        
            write_record_test_file(cfg.current_epoch,postfix,loss_type,record_test,writer)
        te_ave_psnr[epoch] = ave_psnr



    epochs,psnr = zip(*te_ave_psnr.items())
    best_index = np.argmax(psnr)
    best_epoch,best_psnr = epochs[best_index],psnr[best_index]
    print(f"Best Epoch {best_epoch} | Best PSNR {best_psnr} | N: {cfg.N} | Blind: {blind}")
    
    root = Path(f"{settings.ROOT_PATH}/output/noise2aligned/{postfix}/")
    # if cfg.blind: root = root / Path(f"./blind/")
    # else: root = root / Path(f"./nonblind/")
    fn = Path(f"results.csv")

    if not root.exists(): root.mkdir(parents=True)
    path = root / fn
    with open(path,'w') as f:
        f.write("{:d},{:d},{:2.10e},{:d}\n".format(cfg.N,best_epoch,best_psnr,nparams))
    
    save_model(cfg, model, optimizer)

def write_record_losses_file(current_epoch,postfix,loss_type,record_losses):
    root = Path(f"{settings.ROOT_PATH}/output/noise2aligned/{postfix}/")
    if not root.exists(): root.mkdir(parents=True)
    path = root / f"record_losses_{current_epoch}_{loss_type}.csv"
    print(f"Writing record_losses to {path}")
    record_losses.to_csv(path)

def write_record_test_file(current_epoch,postfix,loss_type,record_test):
    root = Path(f"{settings.ROOT_PATH}/output/noise2aligned/{postfix}/")
    if not root.exists(): root.mkdir(parents=True)
    path = root / f"record_test_{current_epoch}_{loss_type}.csv"
    print(f"Writing record_test to {path}")
    record_test.to_csv(path)

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
