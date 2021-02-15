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
    cfg.dataset.name = "cifar10"
    # cfg.dataset.name = "voc"
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
    cfg.dynamic.bool = False

    print("HI")
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
    data,loader = load_dataset(cfg,'denoising')

    checkpoint = cfg.model_path / Path("checkpoint_{}.tar".format(cfg.epochs))
    model = load_model(cfg)
    model = load_model_fp(cfg,model,checkpoint,0)
    # vis_list = ['conv1.single_conv.0.weight','conv1.single_conv.0.bias',
    #             'conv1.single_conv.1.weight','conv1.single_conv.1.bias']
    vis_list = ['conv1.double_conv.0.weight','conv1.double_conv.0.bias',
                'conv1.double_conv.1.weight','conv1.double_conv.1.bias']
    root = Path(f"{settings.ROOT_PATH}/output/n2n/filters/{postfix}/")
    print("root",root)
    if not root.exists(): root.mkdir(parents=True)
    for name,params in model.named_parameters():
        if not (name in vis_list): continue
        name_str = name.replace(".","-") + ".png"
        traj_str = name.replace(".","-") + "_traj" + ".png"
        path = root / Path(name_str)
        path_traj = root / Path(traj_str)
        filters = params.data.detach().cpu()
        # print(name,filters.shape)
        if len(filters.shape) == 1: continue
        NF = filters.shape[0]
        T = filters.shape[1] // 3

        # filters = np.transpose(filters,(1,0,2,3))
        sample = next(iter(loader.tr))[0][0]
        sample += 0.5
        sample.clamp_(0.,1.)
        # print("s",sample.shape)
        for i in range(cfg.N-1):
            print(i*3,(i*3+3))
            print(i,filters[4][i*3:(i*3+3)].view(-1))
        print('a',filters.shape)
        nrow = len(filters)
        filters = torch.cat([filters[:,(3*i):(3*i)+3] for i in range(cfg.N-1)],dim=0)
        print('-->',filters.shape)
        # im = vutils.make_grid(sample,nrow=nrow)
        im = vutils.make_grid(filters,nrow=nrow,padding=1,normalize=True)
        # grids = [vutils.make_grid(filters[i],nrow=8) for i in range(cfg.dynamic.frames)]
        # ims = [[ax.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in grids]
        print('b',im.shape)
        #print(im[:,:,12:16])

        im = np.transpose(im,(1,2,0))
        h,w = im.shape[0:2]
        my_dpi = 1
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w,h)
        ax = plt.Axes(fig,[0.,0.,1.,1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        # fig,ax = plt.subplots(figsize=(h/my_dpi,w/my_dpi))
        ax.imshow(im,aspect='equal',interpolation='none')
        # ax.set_xticklabels([])
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_yticklabels([])
        # print('c',im.shape)
        print(path,im.shape)
        plt.savefig(path,dpi=my_dpi)
        plt.clf()
        plt.cla()
        plt.close("all")
        continue
        
        assert NF*T == filters.shape[0], "Number of filters and timesteps equal."
        if cfg.N < 5: continue
        xgrid_diff = np.arange(T-1)
        xgrid = np.arange(T)
        for f_index in range(NF):
            traj_str = name.replace(".","-") + "_traj" + "_{}".format(f_index) + ".png"
            fig,ax = plt.subplots(figsize=(10,10))
            path_traj = root / Path(traj_str)
            index_over_time = np.arange(f_index,NF*T,NF)
            filter_t = filters[index_over_time]
            # print("f_t",filter_t.shape)
            for channel_index in range(3):
                ftc = filter_t[:,channel_index]
                # print("ftc",ftc.shape)
                for i in range(3):
                    for j in range(3):
                        ftc_ij = ftc[:,i,j]
                        # print("ftc_ij",ftc_ij.shape)
                        ftc_vector = np.ediff1d(ftc_ij)
                        # print(xgrid.shape,ftc_vector.shape)
                        # ax.plot(xgrid_diff,ftc_vector)
                        ax.plot(xgrid,ftc_ij)
            print(path_traj)
            plt.savefig(path_traj)
            plt.clf()
            plt.cla()
            plt.close("all")
                

def run_vis_filters_grid():
    ngpus = 3
    nprocs_per_gpu = 1
    nprocs = ngpus * nprocs_per_gpu
    Sgrid = [50000]
    # Ggrid = [10,25,50,100,150,200]
    Ggrid = [25]
    # Ngrid = [2,3,5,30,20,10,50,100,4]
    # Ngrid = [10,5,3]
    Ngrid = [100,10,5,3]
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
