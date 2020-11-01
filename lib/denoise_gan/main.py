
# python imports
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# pytorch import 
import torch
from torch import nn
import torchvision.utils as vutils
import torch.nn.functional as F

# project code
from datasets import load_dataset
# from learning.test import thtest_denoising as test_loop
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr

# [this folder] project code
from denoise_gan.config import get_cfg,get_args
from .model_io import load_model_gen,load_model_disc
from .optim_io import load_optimizer_gen,load_optimizer_disc
from .sched_io import load_scheduler_gen,load_scheduler_disc

def train_loop(cfg,model_gen,model_disc,optimizer_gen,optimizer_disc,
               criterion,train_loader,epoch,num_epochs,fixed_noise):

    device = cfg.device
    real_label = 1 #cfg.real_label
    fake_label = not real_label
    nz = 100 # cfg.nz
    cfg.log_interval = 10
    cfg.train_gen_interval = 500
    # need: cfg.log_interval,cfg.train_gen_interval

    img_list = []
    D_losses = []
    G_losses = []
    one = torch.FloatTensor([1.]).to(device)
    mone = one * -1.
    idx = 0

    for batch_idx, (noisy_imgs, target) in enumerate(train_loader):
    # for batch_idx, (noisy_imgs, raw_imgs) in enumerate(train_loader):
        # idx += 1
        # if idx > 30: break

        # -- setup --
        N,BS = noisy_imgs.shape[:2]
        p_shape = noisy_imgs.shape[2:]
        p_view = (N*BS,)+p_shape
        n_view = noisy_imgs.shape

        # N,_BS = noisy_imgs.shape[:2]
        # noisy_imgs = noisy_imgs.view(N*_BS,3,32,32)
        # BS = N*_BS

        # ----------------------------
        # (1) Maximize Discriminater
        # ----------------------------
        for p in model_disc.parameters():
            p.requires_grad = True

        model_disc.zero_grad()
        noisy_imgs = noisy_imgs.to(device)

        # -- (i) real images
        noisy_imgs = noisy_imgs.view(p_view)
        output = model_disc(noisy_imgs).view(-1)
        err_disc_real = output.mean(0).view(1)
        err_disc_real.backward(one)
        D_x = output.mean().item()

        # -- (ii) fake images
        fake = model_gen(noisy_imgs.view(p_view))
        output = model_disc(fake.detach()).view(-1)
        err_disc_fake = output.mean(0).view(1)
        err_disc_fake.backward(mone)
        D_G_z1 = output.mean().item()

        error_disc = err_disc_real - err_disc_fake
        optimizer_disc.step()
        
        for p in model_disc.parameters():
            p.data.clamp_(-0.01, 0.01)
        # if (batch_idx % 2) == 0 and batch_idx > 0: 
        #     optimizer_disc.step()

        # -----------------------
        # (2) Maximize Generator
        # -----------------------
        for p in model_disc.parameters():
            p.requires_grad = False

        error_gen = (error_disc - error_disc) 
        D_G_z2 = (D_G_z1 - D_G_z1)
        if ((batch_idx % 5) == 0 and batch_idx > 0 and epoch > 1) or ((batch_idx % 5) == 0 and batch_idx > 0):
            output = model_disc(fake).view(-1)
            error_gen = output.mean(0).view(1)
            error_gen.backward(one)
            D_G_z2 = output.mean().item()
            optimizer_gen.step()

        if (batch_idx % cfg.log_interval) == 0:
            print("[%d/%d][%d/%d] Loss_D: %.4f\tLoss_G %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"%
                  (epoch, num_epochs, batch_idx, len(train_loader),error_disc.item(),error_gen.item(),D_x,D_G_z1,D_G_z2))

        D_losses.append(error_disc)
        G_losses.append(error_gen)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (batch_idx % cfg.train_gen_interval == 0) or ((epoch == num_epochs-1) and (batch_idx == len(train_loader)-1)):
            with torch.no_grad():
                fake = model_gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=16))

    return D_losses,G_losses,img_list
        
def test_denoising(cfg, model, test_loader):
    model.eval()
    test_loss = 0

    idx = 0
    with torch.no_grad():
        for noisy_imgs, raw_img in tqdm(test_loader):
            set_loss = 0
            N,BS = noisy_imgs.shape[:2]
            p_shape = noisy_imgs.shape[2:]

            noisy_imgs = noisy_imgs.cuda(non_blocking=True)
            raw_img = raw_img.cuda(non_blocking=True)

            noisy_imgs = noisy_imgs.cuda(non_blocking=True)
            noisy_imgs = noisy_imgs.view((N*BS,)+p_shape)
            dec_imgs = model(noisy_imgs)
            dec_no_rescale = dec_imgs
            dec_imgs = rescale_noisy_image(dec_imgs)

            dshape = (N,BS,) + p_shape
            dec_imgs = dec_imgs.reshape(dshape)
            raw_img = raw_img.expand(dshape)

            if idx == 10:
                print('dec_no_rescale',dec_no_rescale.mean(),dec_no_rescale.min(),dec_no_rescale.max())
                print('noisy',noisy_imgs.mean(),noisy_imgs.min(),noisy_imgs.max())
                print('dec',dec_imgs.mean(),dec_imgs.min(),dec_imgs.max())
                print('raw',raw_img.mean(),raw_img.min(),raw_img.max())

            loss = F.mse_loss(raw_img,dec_imgs).item()
            if cfg.test_with_psnr: loss = mse_to_psnr(loss)
            test_loss += loss
            idx += 1
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:2.3e}\n'.format(test_loss))
    return test_loss



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
