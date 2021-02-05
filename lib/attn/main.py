

# -- python imports --
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import datetime

# -- pytorch import --
import torch
from torch import nn
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# -- project code --
import settings
from datasets import load_dataset
# from learning.test import thtest_denoising as test_loop
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from learning.utils import save_model

# -- [this folder] project code --
from .config import get_cfg,get_args
from .model_io import load_model,load_model_fp
from .optim_io import load_optimizer
from .sched_io import load_scheduler
from .learn import train_loop,test_loop

def run_me():
    args = get_args()
    args.name = "default_attn_16_lowNoise"
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
    cfg.batch_size = 48

    cfg.blind = False
    cfg.N = 3
    cfg.dynamic.frames = cfg.N
    cfg.noise_type = 'g'
    cfg.noise_params['g']['stddev'] = 25.
    noise_level = 5.
    cfg.init_lr = 1e-3
    cfg.unet_channels = 3
    cfg.input_N = cfg.N-1
    cfg.epochs = 1000
    cfg.log_interval = int(int(50000 / cfg.batch_size) / 20) # NumOfBatches / Const = NumOfPrints
    cfg.save_epoch = 50

    # -- attn parameters --
    cfg.patch_sizes = [48,48]
    cfg.d_model_attn = 3 # 512

    cfg.dataset.name = "voc"
    cfg.dataset.bw = False
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 2
    cfg.dynamic.frame_size = 48
    cfg.dynamic.total_pixels = 6
    cfg.load = False

    # left; 5 total pix; 10 in 8 out 16 imsize; bs 64; gpuid 1; 26 PSNR @ 30 and 190 epochs
    # right; 20 total pix; 16 in 8 out 16 imsize; bs 12; gpuid 0; 21 PSNR @ 70 epochs
    # 16_2_20 ; 27.59 psnr 16 in? 16 out? 16 imsize
    # 10, 8 was also promising

    # -- model type --
    cfg.model_version = "v4"

    # -- input noise for learning --
    cfg.input_noise = True
    cfg.input_noise_level = 25./255
    cfg.input_with_middle_frame = True
    if cfg.input_with_middle_frame:
        cfg.input_N = cfg.N

    # -- learning --
    cfg.spoof_batch = False
    cfg.spoof_batch_size = 1

    torch.cuda.set_device(gpuid)

    # cfg.batch_size = 256
    cfg.init_lr = 5e-5
    print(cfg.batch_size,cfg.init_lr)

    # -- load model -- 
    model = load_model(cfg)
    # inputs = torch.rand(5,10,3,32,32).to(cfg.device)
    # outputs = torch.rand(10,3,32,32).to(cfg.device)
    # encode = model(inputs,outputs)
    optimizer = load_optimizer(cfg,model)
    # -- load data
    data,loader = load_dataset(cfg,'dynamic')
    # data,loader = load_dataset(cfg,'denoising')


    blind = "blind" if cfg.blind else "nonblind"
    # print(grid_idx,blind,cfg.N,Ggrid[G_grid_idx],gpuid)
    # if blind == "nonblind": return 
    dynamic_str = "dynamic_input_noise" if cfg.input_noise else "dynamic"
    if cfg.input_with_middle_frame: dynamic_str += "_wmf"
    postfix = Path(f"./{cfg.model_version}/{dynamic_str}/{cfg.dynamic.frame_size}_{cfg.dynamic.ppf}_{cfg.dynamic.total_pixels}/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    print(f"model dir [{postfix}]")
    cfg.model_path = cfg.model_path / postfix
    cfg.optim_path = cfg.optim_path / postfix
    if not cfg.model_path.exists(): cfg.model_path.mkdir(parents=True)
    if not cfg.optim_path.exists(): cfg.optim_path.mkdir(parents=True)
    
    checkpoint = cfg.model_path / Path("checkpoint_{}.tar".format(cfg.epochs))
    print(checkpoint)
    # if checkpoint.exists(): return

    print("N: {} | Noise Level: {}".format(cfg.N,cfg.noise_params['g']['stddev']))
    torch.cuda.set_device(gpuid)

    # load criterion
    criterion = None

    if cfg.load:
        fp = cfg.model_path / Path("checkpoint_30.tar")
        model = load_model_fp(cfg,model,fp,0)

    cfg.current_epoch = 0
    te_ave_psnr = {}
    test_before = False
    # -- test before --
    if test_before:
        ave_psnr = test_loop(cfg,model,criterion,loader.te,-1)
        print("PSNR before training {:2.3e}".format(ave_psnr))
        return 

    # -- load trained model --
    if checkpoint.exists() and cfg.load:
        model = load_model_fp(cfg,model,checkpoint,gpuid)
        print("Loaded model.")
        cfg.current_epoch = cfg.epochs
        
    # -- summary writer --
    dsname = cfg.dataset.name
    cfg.summary_log_dir = Path(f"{settings.ROOT_PATH}/runs/attn/{dsname}/{cfg.exp_name}/{postfix}")
    datetime_now = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    writer_dir = cfg.summary_log_dir / Path(datetime_now)
    writer = SummaryWriter(writer_dir)
    print(f"Summary writer @ [{writer_dir}]")

    # -- training loop --
    for epoch in range(cfg.current_epoch,cfg.epochs):

        losses = train_loop(cfg,model,optimizer,criterion,loader.tr,epoch)
        writer.add_scalar("Loss/train", losses, epoch)
        ave_psnr = test_loop(cfg,model,criterion,loader.te,epoch)
        writer.add_scalar("Loss/test-ave-psnr", ave_psnr, epoch)
        te_ave_psnr[epoch] = ave_psnr
        cfg.current_epoch += 1
        writer.flush()
        if (epoch % cfg.save_epoch) == 0:
            save_model(cfg, model, optimizer)

    epochs,psnr = zip(*te_ave_psnr.items())
    best_index = np.argmax(psnr)
    best_epoch,best_psnr = epochs[best_index],psnr[best_index]
    
    root = Path(f"{settings.ROOT_PATH}/output/attn/{postfix}/")
    # if cfg.blind: root = root / Path(f"./blind/")
    # else: root = root / Path(f"./nonblind/")
    fn = Path(f"results.csv")

    if not root.exists(): root.mkdir(parents=True)
    path = root / fn
    with open(path,'w') as f:
        f.write("{:d},{:d},{:2.10e},{:d}\n".format(cfg.N,best_epoch,best_psnr,nparams))
    
    save_model(cfg, model, optimizer)

