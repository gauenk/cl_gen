"""
Visualize convolution filters from unet

"""

# python imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# pytorch imports
import torch
import torchvision.utils as vutils
import torch.multiprocessing as mp

# project imports
import settings
from datasets import load_dataset

# [this folder] project code
from .config import get_cfg,get_args
from .model_io import load_model,load_model_fp

def run_vis_filters(rank=0,Sgrid=1,Ngrid=1,nNgrid=1,Ggrid=1,nGgrid=1,ngpus=3,idx=0):

    args = get_args()
    args.name = "default"
    cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.use_apex = False
    gpuid = rank % ngpus # set gpuid
    cfg.device = f"cuda:{gpuid}"

    grid_idx = idx*(1*ngpus)+rank
    B_grid_idx = (grid_idx % 2)
    N_grid_idx = ( grid_idx // 2 ) % nNgrid
    G_grid_idx = grid_idx // (nNgrid * 2) % nGgrid
    S_grid_idx = grid_idx // (nGgrid * nNgrid * 2) 
    # print(grid_idx,B_grid_idx,N_grid_idx,G_grid_idx,S_grid_idx)

    cfg.use_collate = True
    # cfg.dataset.download = False
    # cfg.cls = cfg
    cfg.S = Sgrid[S_grid_idx]
    # cfg.dataset.name = "cifar10"
    cfg.dataset.name = "voc"
    cfg.blind = (B_grid_idx == 0)
    cfg.N = Ngrid[N_grid_idx]
    cfg.dynamic.frames = cfg.N
    cfg.noise_type = 'g'
    cfg.noise_params['g']['stddev'] = Ggrid[G_grid_idx]
    noise_level = Ggrid[G_grid_idx]
    cfg.batch_size = 16
    cfg.init_lr = 1e-3
    cfg.unet_channels = 3
    # if cfg.blind: cfg.input_N = cfg.N - 1
    # else: cfg.input_N = cfg.N
    cfg.input_N = cfg.N-1
    cfg.epochs = 30
    cfg.log_interval = int(int(50000 / cfg.batch_size) / 100)
    cfg.dynamic.frames = cfg.N
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 2
    cfg.dynamic.frame_size = 128
    cfg.dynamic.total_pixels = 20
    cfg.num_workers = 4
    torch.manual_seed(0)
    cfg.set_worker_seed = True
    

    # if cfg.N != 10 or (not cfg.blind): return
    blind = "blind" if cfg.blind else "nonblind"
    # if cfg.dynamic.bool:
    #     postfix = Path(f"./dynamic/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    # else:
    #     postfix = Path(f"./{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    postfix = Path(f"./dynamic/{cfg.dynamic.frame_size}_{cfg.dynamic.ppf}/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")

    cfg.model_path = cfg.model_path / postfix
    cfg.optim_path = cfg.optim_path / postfix
    if not cfg.model_path.exists(): cfg.model_path.mkdir(parents=True)
    if not cfg.optim_path.exists(): cfg.optim_path.mkdir(parents=True)
    torch.cuda.set_device(gpuid)
    # data,loader = load_dataset(cfg,'denoising')
    data,loader = load_dataset(cfg,'dynamic')

    checkpoint = cfg.model_path / Path("checkpoint_{}.tar".format(cfg.epochs))
    model = load_model(cfg)
    model = load_model_fp(cfg,model,checkpoint,0)
    
    noisy,raw = next(iter(loader.te))

    ave = torch.mean(noisy,dim=0)

    noisy += 0.5
    noisy.clamp_(0,1.)

    ave += 0.5
    ave.clamp_(0.,1.)
    ave = ave.expand(noisy.shape)
        
    # -- reconstruct image -- 
    middle = cfg.N // 2
    input_order = np.r_[np.arange(0,middle),np.arange(middle+1,cfg.N)]
    burst = torch.cat([noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
    rec = model(burst).detach().cpu()
    print("rec stats: ",rec.min(),rec.max(),rec.mean())
    rec += 0.5
    rec.clamp_(0.,1.)

    rec = rec.expand(noisy.shape)
    raw = raw.expand(noisy.shape)

    # images = torch.cat([noisy,raw,rec],dim=1)
    images = torch.cat([noisy,raw,rec,ave],dim=1)
    REP = images.shape[1] // noisy.shape[1]
    
    plot_batch = True
    plot_pick = True
    plot_pick_movie = False
    if plot_batch:
        fig,ax = plt.subplots(figsize=(10,4))
        ax.set_axis_off()
        grids = [vutils.make_grid(images[i],nrow=cfg.batch_size,normalize=False)
             for i in range(cfg.dynamic.frames)]
        ims = [[ax.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in grids]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)
        path = f"{settings.ROOT_PATH}/rec_voc_{cfg.blind}_{cfg.N}.mp4"
        ani.save(path, writer=writer)
        print(f"Wrote to {path}")
    if plot_pick:

        pick_idx = 4
        path = f"{settings.ROOT_PATH}/rec_voc_{cfg.blind}_{cfg.N}_{pick_idx}"
        sub_idx = np.arange(pick_idx,cfg.batch_size*REP,cfg.batch_size)
        subimg = images[:,sub_idx]

        if plot_pick_movie:
            fig,ax = plt.subplots(figsize=(10,4))
            ax.set_axis_off()
            path += ".mp4"

            grids = [vutils.make_grid(subimg[i],nrow=cfg.batch_size,normalize=False)
                 for i in range(cfg.dynamic.frames)]
            ims = [[ax.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in grids]
            ani = animation.ArtistAnimation(fig, ims, interval=1000,
                                            repeat_delay=1000, blit=True)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(path, writer=writer)
        else:
            fig,ax = plt.subplots(figsize=(10,4))
            ax.set_axis_off()
            path += ".png"

            grid = vutils.make_grid(subimg[0],nrow=REP,normalize=False)
            ims = ax.imshow(np.transpose(grid,(1,2,0)), animated=True)
            plt.savefig(path)

        print(f"Wrote to {path}")
    plt.close("all")
    

def run_vis_rec_grid():
    ngpus = 3
    nprocs_per_gpu = 1
    nprocs = ngpus * nprocs_per_gpu
    Sgrid = [50000]
    # Ggrid = [10,25,50,100,150,200]
    Ggrid = [25]
    # Ngrid = [2,3,5,30,20,10,50,100,4]
    # Ngrid = [10,5,3]
    Ngrid = [10,5,3]
    nNgrid = len(Ngrid)
    nGgrid = len(Ggrid)
    te_losses = dict.fromkeys(Ngrid)
    num_of_grids = 2 * len(Sgrid) * len(Ggrid) * len(Ngrid) // nprocs
    for idx in range(num_of_grids):
        # for gpuid in range(ngpus):
        #     run_me(gpuid,Sgrid,Ngrid,nNgrid,Ggrid,nGgrid,ngpus,idx)
        r = mp.spawn(run_vis_filters, nprocs=nprocs,
                     args=(Sgrid,Ngrid,nNgrid,Ggrid,nGgrid,ngpus,idx))

    

if __name__ == "__main__":
    print("HI")
    main()
