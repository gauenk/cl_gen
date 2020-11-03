"""
Visualize convolution filters from unet

"""

# python imports
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# pytorch imports
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.multiprocessing as mp

# project imports
import settings
from datasets import load_dataset
from pyutils.misc import mse_to_psnr

# [this folder] project code
from .config import get_cfg,get_args
from .model_io import load_model,load_model_fp
from .dncnn import get_postfix_str as get_postfix_str_dncnn
from .n2n_main import get_postfix_str as get_postfix_str_n2n

def test_methods(cfg,loader,ours,n2n,dncnn):
    BS = cfg.batch_size
    ours_psnr,n2n_psnr,dncnn_psnr = 0,0,0
    ours = ours.to(cfg.device)
    n2n = n2n.to(cfg.device)
    dncnn = dncnn.to(cfg.device)
    for idx,(noisy,res,raw) in enumerate(loader.te):

        noisy = noisy.cuda(non_blocking=True)
        # print("raw stats",raw.min().item(),raw.max().item(),raw.mean().item())
        # print("noisy stats: ",noisy.min().item(),noisy.max().item(),noisy.mean().item())
        

        # -- ours --
        middle = cfg.N // 2
        input_order = np.r_[np.arange(0,middle),np.arange(middle+1,cfg.N)]
        burst = torch.cat([noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
        rec = ours(burst).detach().cpu() + 0.5
        loss = F.mse_loss(raw,rec,reduction='none').reshape(BS,-1)
        loss = torch.mean(loss,1).detach().cpu().numpy()
        ours_psnr += np.mean(mse_to_psnr(loss))
        

        # -- n2n --
        n2n_img = n2n(noisy[cfg.N//2]).detach().cpu() + 0.5
        loss = F.mse_loss(raw,n2n_img,reduction='none').reshape(BS,-1)
        loss = torch.mean(loss,1).detach().cpu().numpy()
        n2n_psnr += np.mean(mse_to_psnr(loss))
        # print("n2n: {:2.1f}".format(np.mean(mse_to_psnr(loss))))

        # -- dncnn --
        dncnn_res = dncnn(noisy[cfg.N//2]).detach()
        dncnn_img = noisy[cfg.N//2] + 0.5 - dncnn_res
        loss = F.mse_loss(raw,dncnn_img.cpu(),reduction='none').reshape(BS,-1)
        loss = torch.mean(loss,1).detach().cpu().numpy()
        dncnn_psnr += np.mean(mse_to_psnr(loss))
        # print("dncnn: {:2.1f}".format(np.mean(mse_to_psnr(loss))))

        if (idx % 20) == 0 and idx > 0:
            print("[{:d}/{:d}]: [Ours]: {:2.1f} [N2N]: {:2.1f} [DnCNN]: {:2.1f}".format(idx,len(loader.te),ours_psnr/idx,dncnn_psnr/idx,n2n_psnr/idx))

    ours_psnr /= len(loader.te)
    n2n_psnr /= len(loader.te)
    dncnn_psnr /= len(loader.te)
    print("[Ours] PSNR: {:2.1f}".format(ours_psnr))
    print("[N2N] PSNR: {:2.1f}".format(n2n_psnr))
    print("[DnCNN] PSNR: {:2.1f}".format(dncnn_psnr))

    n2n = n2n.to('cpu')
    dncnn = dncnn.to('cpu')


def print_weight(n2n):
    for name,params in n2n.named_parameters():
        print("{:s}: {:2.1e}".format(name,torch.mean(params.data[0])))
        return

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
    cfg.dynamic.reset_seed = True
    cfg.num_workers = 4
    torch.manual_seed(0)
    cfg.set_worker_seed = True
    BS = cfg.batch_size
    

    # if cfg.N != 10 or (not cfg.blind): return
    blind = "blind" if cfg.blind else "nonblind"
    # if cfg.dynamic.bool:
    #     postfix = Path(f"./dynamic/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    # else:
    #     postfix = Path(f"./{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    postfix = Path(f"./dynamic/{cfg.dynamic.frame_size}_{cfg.dynamic.ppf}_{cfg.dynamic.total_pixels}/{cfg.S}/{blind}/{cfg.N}/{noise_level}/")
    model_base = cfg.model_path
    cfg.model_path = cfg.model_path / postfix
    cfg.optim_path = cfg.optim_path / postfix
    if not cfg.model_path.exists(): cfg.model_path.mkdir(parents=True)
    if not cfg.optim_path.exists(): cfg.optim_path.mkdir(parents=True)
    torch.cuda.set_device(gpuid)
    # data,loader = load_dataset(cfg,'denoising')
    cfg.dynamic.frame_size = 128
    data,loader = load_dataset(cfg,'dynamic')

    # -- [ours] checkpoint load model -- 
    checkpoint = cfg.model_path / Path("checkpoint_{}.tar".format(cfg.epochs))
    model = load_model(cfg)
    print(f"[ours]: {checkpoint}")
    model = load_model_fp(cfg,model,checkpoint,0)
    model = model.eval()
    
    # -- [dncnn] checkpoint load model -- 
    d_cfg = copy.deepcopy(cfg)
    d_cfg.N = 2
    d_cfg.input_N = 1
    d_cfg.dynamic.ppf = 0
    d_cfg.dynamic.total_pixels = 0
    d_cfg.dynamic.frame_size = 128
    d_cfg.dynamic.nframes = cfg.N
    dncnn_postfix = get_postfix_str_dncnn(d_cfg,"blind",noise_level)
    print(dncnn_postfix)
    d_cfg.model_path = model_base / dncnn_postfix
    checkpoint = d_cfg.model_path / Path("checkpoint_{}.tar".format(d_cfg.epochs))
    dncnn = load_model(d_cfg)
    dncnn = load_model_fp(d_cfg,dncnn,checkpoint,0)
    dncnn = dncnn.eval()

    # -- [n2n] checkpoint load model -- 
    d_cfg = copy.deepcopy(cfg)
    d_cfg.N = 2
    d_cfg.input_N = 1
    d_cfg.dynamic.ppf = 0
    d_cfg.dynamic.total_pixels = 0
    d_cfg.dynamic.frame_size = 128
    d_cfg.dynamic.nframes = cfg.N
    n2n_postfix = get_postfix_str_n2n(d_cfg,"blind",noise_level)
    d_cfg.model_path = model_base / n2n_postfix
    checkpoint = d_cfg.model_path / Path("checkpoint_{}.tar".format(d_cfg.epochs))
    n2n = load_model(d_cfg)
    # print_weight(n2n)
    n2n = load_model_fp(d_cfg,n2n,checkpoint,gpuid)
    # print_weight(n2n)
    n2n = n2n.eval()

    # -- load sample -- 
    test_methods(cfg,loader,model,n2n,dncnn)
    noisy,res,raw = next(iter(loader.te))

    # -- compute average --
    ave = torch.mean(noisy,dim=0)
    ave += 0.5
    loss = F.mse_loss(raw,ave,reduction='none').reshape(BS,-1)
    loss = torch.mean(loss,1).detach().cpu().numpy()
    ave_psnr = np.mean(mse_to_psnr(loss))
    ave.clamp_(0.,1.)

    # -- [burst-n2n] reconstruct image -- 
    middle = cfg.N // 2
    input_order = np.r_[np.arange(0,middle),np.arange(middle+1,cfg.N)]
    burst = torch.cat([noisy[input_order[x]] for x in range(cfg.input_N)],dim=1)
    print("[0] burst stats: ",burst.min().item(),burst.max().item(),burst.mean().item())
    rec = model(burst).detach().cpu() + 0.5
    print("[a] rec stats: ",rec.min().item(),rec.max().item(),rec.mean().item())
    print("[b] raw stats: ",raw.min().item(),raw.max().item(),raw.mean().item())
    loss = F.mse_loss(raw,rec,reduction='none').reshape(BS,-1)
    loss = torch.mean(loss,1).detach().cpu().numpy()
    ours_psnr = np.mean(mse_to_psnr(loss))
    print("[Ours] PSNR: {:2.1f}".format(ours_psnr))
    rec.clamp_(0.,1.)



    # print(rec.shape,torch.mean(rec.view(cfg.batch_size,-1),dim=1))
    # rec -= torch.min(rec.view(cfg.batch_size,-1),dim=1)[0][:,None,None,None].expand(rec.shape)

    # print(rec.shape,torch.mean(rec.view(cfg.batch_size,-1),dim=1))
    # print(torch.max(rec.view(cfg.batch_size,-1),dim=1)[0])
    # rec /= torch.max(rec.view(cfg.batch_size,-1),dim=1)[0][:,None,None,None].expand(rec.shape)
    # print("[c] rec stats: ",rec.min().item(),rec.max().item(),rec.mean().item())
    # print(rec.shape,torch.mean(rec.view(cfg.batch_size,-1),dim=1))

    # -- [n2n] reconstruct image -- 
    middle = cfg.N // 2
    middle_frame = noisy[middle]
    print(middle_frame.shape)
    rec_n2n = n2n(middle_frame).detach().cpu() + 0.5
    loss = F.mse_loss(raw,rec_n2n,reduction='none').reshape(BS,-1)
    loss = torch.mean(loss,1).detach().cpu().numpy()
    n2n_psnr = np.mean(mse_to_psnr(loss))
    print("[n2n] rec_n2n stats: ",rec_n2n.min().item(),rec_n2n.max().item(),rec_n2n.mean().item())
    rec_n2n.clamp_(0.,1.)
    print("[N2N] PSNR: {:2.1f}".format(n2n_psnr))


    # -- [dncnn] reconstruct image -- 
    middle = cfg.N // 2
    middle_frame = noisy[middle]
    print(middle_frame.shape)
    rec_res = dncnn(middle_frame).detach().cpu()
    rec_dncnn = middle_frame + 0.5 - rec_res
    print("[dncnn] rec_dncnn stats: ",rec_dncnn.min().item(),rec_dncnn.max().item(),rec_dncnn.mean().item())
    loss = F.mse_loss(raw,rec_dncnn,reduction='none').reshape(BS,-1)
    loss = torch.mean(loss,1).detach().cpu().numpy()
    dncnn_psnr = np.mean(mse_to_psnr(loss))    
    print("[dncnn] rec_dncnn stats: ",rec_dncnn.min().item(),rec_dncnn.max().item(),rec_dncnn.mean().item())
    rec_dncnn.clamp_(0.,1.)
    print("[DnCNN] PSNR: {:2.1f}".format(dncnn_psnr))
    
    # -- prepare noisy images for visualization --
    noisy += 0.5
    noisy.clamp_(0,1.)


    # -- create image plot --
    # print(noisy.min(),noisy.max(),noisy.mean())
    # print(raw.min(),raw.max(),raw.mean())
    # print(ave.min(),ave.max(),ave.mean())
    # print(rec_dncnn.min(),rec_dncnn.max(),rec_dncnn.mean())
    raw = raw.expand(noisy.shape)
    rec = rec.expand(noisy.shape)
    rec_n2n = rec_n2n.expand(noisy.shape)
    rec_dncnn = rec_dncnn.expand(noisy.shape)
    ave = ave.expand(noisy.shape)

    images = torch.cat([noisy,raw,ave,rec,rec_n2n,rec_dncnn],dim=1)
    REP = images.shape[1] // noisy.shape[1]

    # -- plotting -- 
    plot_batch = True
    plot_pick = True
    plot_pick_movie = False
    path = f"{settings.ROOT_PATH}/rec_voc_{blind}_{cfg.N}_{cfg.dynamic.total_pixels}_{cfg.dynamic.frame_size}"
    if plot_batch:
        fig,ax = plt.subplots(figsize=(10,4))
        ax.set_axis_off()
        grids = [vutils.make_grid(images[i],nrow=cfg.batch_size,normalize=False)
             for i in range(cfg.dynamic.frames)]
        ims = [[ax.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in grids]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(path+".mp4", writer=writer)
        print(f"Wrote to {path}")
    if plot_pick:

        pick_idx = 3
        sub_idx = np.arange(pick_idx,cfg.batch_size*REP,cfg.batch_size)
        subimg = images[:,sub_idx]

        if plot_pick_movie:
            fig,ax = plt.subplots(figsize=(8,3))
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
            path += ".png"
            fig,ax = plt.subplots(figsize=(8,3))
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(np.arange(REP)*(cfg.dynamic.frame_size+1) + cfg.dynamic.frame_size*.45)
            labels = ["Noisy","Clean","Ave","Ours","N2N","DnCNN"]
            labels[2] += " {:2.1f}".format(ave_psnr)
            labels[3] += " {:2.1f}".format(ours_psnr)
            labels[4] += " {:2.1f}".format(n2n_psnr)
            labels[5] += " {:2.1f}".format(dncnn_psnr)
            ax.set_xticklabels(labels)

            grid = vutils.make_grid(subimg[0],nrow=REP,normalize=False)
            ims = ax.imshow(np.transpose(grid,(1,2,0)), animated=True)
            plt.savefig(path,transparent=True)

        print(f"Wrote to {path}")
    plt.close("all")
    

def run_vis_rec_grid():
    ngpus = 2
    nprocs_per_gpu = 1
    nprocs = ngpus * nprocs_per_gpu
    Sgrid = [50000]
    # Ggrid = [10,25,50,100,150,200]
    Ggrid = [25]
    # Ngrid = [2,3,5,30,20,10,50,100,4]
    # Ngrid = [10,5,3]
    Ngrid = [20]
    nNgrid = len(Ngrid)
    nGgrid = len(Ggrid)
    te_losses = dict.fromkeys(Ngrid)
    num_of_grids = 2 * len(Sgrid) * len(Ggrid) * len(Ngrid) // nprocs + 1
    for idx in range(num_of_grids):
        # for gpuid in range(ngpus):
        #     run_me(gpuid,Sgrid,Ngrid,nNgrid,Ggrid,nGgrid,ngpus,idx)
        r = mp.spawn(run_vis_filters, nprocs=nprocs,
                     args=(Sgrid,Ngrid,nNgrid,Ggrid,nGgrid,ngpus,idx))

    

if __name__ == "__main__":
    print("HI")
    main()
