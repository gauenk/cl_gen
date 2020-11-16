
# -- python imports --
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# -- pytorch import --
import torch
from torch import nn
import torchvision.utils as vutils
import torch.nn.functional as F

# -- project code --
from datasets import load_dataset
# from learning.test import thtest_denoising as test_loop
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr

# -- [this folder] project code --
from denoise_gan.config import get_cfg,get_args
from .model_io import load_model
from .optim_io import load_optimizer
from .sched_io import load_scheduler

def run_me():
    args = get_args()
    cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.use_apex = False
    gpuid = 2 # set gpuid
    cfg.device = f"cuda:{gpuid}"
    cfg.N = 2
    cfg.use_collate = True
    cfg.dataset.download = False
    cfg.cls = cfg
    cfg.batch_size = 256
    
    torch.cuda.set_device(gpuid)

    # cfg.batch_size = 256
    cfg.init_lr = 5e-5
    print(cfg.batch_size,cfg.init_lr)

    model_gen = load_model_gen(cfg)
    optimizer_gen = load_optimizer_gen(cfg,model_gen)
    scheduler_gen = load_scheduler_gen(cfg,model_gen,optimizer_gen)

    model_disc = load_model_disc(cfg)
    optimizer_disc = load_optimizer_disc(cfg,model_disc)
    schduler_disc = load_scheduler_disc(cfg,model_gen,optimizer_disc)

    data,loader = load_dataset(cfg,'denoising')
    criterion = nn.BCELoss()

    num_epochs = 50
    cfg.nz = 100
    fixed_noise,_ = next(iter(loader.val))
    p_shape = fixed_noise.shape[2:]
    fixed_noise = fixed_noise.view((cfg.N*cfg.batch_size,)+p_shape).to(cfg.device)


    D_losses = []
    G_losses = []
    img_list = []
    # test_denoising(cfg, model_gen, loader.te)
    for epoch in range(num_epochs):
        losses = train_loop(cfg,model_gen,model_disc,optimizer_gen,optimizer_disc,
                            criterion,loader.tr,epoch,num_epochs,fixed_noise)
        D_losses.extend(losses[0])
        G_losses.extend(losses[1])
        img_list.extend(losses[2])
        denoise_loss = test_denoising(cfg, model_gen, loader.te)
        # print(denoise_loss)

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./gan_losses_{gpuid}.png")

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(f"./gan_{gpuid}.mp4", writer=writer)
