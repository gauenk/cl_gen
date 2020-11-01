
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
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr,normalize_image_to_zero_one

# [this folder] project code
from denoise_gan.config import get_cfg,get_args
from .model_io import load_model_gen,load_model_disc,load_model_simclr
from .optim_io import load_optimizer_gen,load_optimizer_disc
from .sched_io import load_scheduler_gen,load_scheduler_disc

def train_loop(cfg,model_gen,model_disc,optimizer_gen,optimizer_disc,
               model_simclr,criterion,train_loader,epoch,num_epochs,fixed_noise):

    device = cfg.device
    real_label = 1 #cfg.real_label
    fake_label = not real_label
    nz = 100 # cfg.nz
    cfg.log_interval = 3
    cfg.train_gen_interval = 500
    # need: cfg.log_interval,cfg.train_gen_interval
    model_simclr.eval()
    for p in model_simclr.parameters():
        p.requires_grad = False

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
        dshape = (N,BS,) + p_shape
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
        simclr_emb,_ = model_simclr(noisy_imgs)
        simclr_emb = simclr_emb.reshape(N*BS,2,32,32)
        output = model_disc(simclr_emb.detach()).view(-1)
        err_disc_real = output.mean(0).view(1)
        err_disc_real.backward(one)
        D_x = output.mean().item()

        # -- (ii) fake images
        fake = model_gen(noisy_imgs.view(p_view))
        simclr_emb,_ = model_simclr(fake.view(n_view))
        simclr_emb = simclr_emb.reshape(N*BS,2,32,32)
        output = model_disc(simclr_emb.detach()).view(-1)
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
        # if ((batch_idx % 15) == 0 and batch_idx > 0 and epoch > 10) or ((batch_idx % 20) and batch_idx > 0):
        if ((batch_idx % 5) == 0 and batch_idx > 0 and epoch > 10) or ((batch_idx % 5) and batch_idx > 0):
        # if ((batch_idx % 5) == 0 and batch_idx > 0 and epoch > 10) or ((batch_idx % 10) == 0 and batch_idx > 0):
            model_gen.zero_grad()
            output = model_disc(simclr_emb).view(-1)
            error_gen = output.mean(0).view(1)
            # error_gen.backward(one)

            D_G_z2 = output.mean().item()
            gen_loss = error_gen

            # include reconstruction loss.
            if epoch < 10:
                offset_idx = [(i+1)%N for i in range(N)]
                noisy_imgs = noisy_imgs.view(dshape)
                fake = fake.view(dshape)
                offset_idx = [(i+1)%N for i in range(N)]
                fake_offset = fake[offset_idx]
                rec_loss = F.mse_loss(fake_offset,noisy_imgs)
                gen_loss += (10-epoch)/10.*rec_loss
            # rec_loss.backward()

            # print(error_gen.shape,rec_loss.shape)
            gen_loss.backward(one)

            optimizer_gen.step()

        if (batch_idx % cfg.log_interval) == 0:
            print("[%d/%d][%d/%d] Loss_D: %.4f\tLoss_G %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"%
                  (epoch, num_epochs, batch_idx, len(train_loader),error_disc.item(),
                   error_gen.item(),D_x,D_G_z1,D_G_z2))

        D_losses.append(error_disc)
        G_losses.append(error_gen)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (batch_idx % cfg.train_gen_interval == 0) or ((epoch == num_epochs-1) and (batch_idx == len(train_loader)-1)):
            with torch.no_grad():
                fake = model_gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=16))

    return D_losses,G_losses,img_list
        
def test_denoising(cfg, model, test_loader,epoch,num_epochs):
    model.eval()
    rigid_loss = 0
    test_loss = 0

    idx = 0
    with torch.no_grad():
        for noisy_imgs, raw_img in tqdm(test_loader):
            set_loss = 0
            N,BS = noisy_imgs.shape[:2]
            p_shape = noisy_imgs.shape[2:]
            bshape = (N*BS,)+p_shape
            dshape = (N,BS,) + p_shape

            noisy_imgs = noisy_imgs.cuda(non_blocking=True)
            raw_img = raw_img.cuda(non_blocking=True)

            noisy_imgs = noisy_imgs.cuda(non_blocking=True)
            noisy_imgs = noisy_imgs.view((N*BS,)+p_shape)
            dec_imgs = model(noisy_imgs).detach()
            dec_no_rescale = dec_imgs
            dec_imgs = rescale_noisy_image(dec_imgs.clone())
            rigid_nmlz_imgs = normalize_image_to_zero_one(dec_imgs.clone())

            rigid_nmlz_imgs = rigid_nmlz_imgs.reshape(dshape)
            dec_imgs = dec_imgs.reshape(dshape)
            raw_img = raw_img.expand(dshape)

            if idx == 10:
                print('dec_no_rescale',dec_no_rescale.mean(),dec_no_rescale.min(),dec_no_rescale.max())
                print('noisy',noisy_imgs.mean(),noisy_imgs.min(),noisy_imgs.max())
                print('dec',dec_imgs.mean(),dec_imgs.min(),dec_imgs.max())
                print('raw',raw_img.mean(),raw_img.min(),raw_img.max())

            r_loss = F.mse_loss(raw_img,rigid_nmlz_imgs).item()
            if cfg.test_with_psnr: r_loss = mse_to_psnr(r_loss)
            rigid_loss += r_loss

            loss = F.mse_loss(raw_img,dec_imgs).item()
            if cfg.test_with_psnr: loss = mse_to_psnr(loss)
            test_loss += loss
            idx += 1
    test_loss /= len(test_loader)
    rigid_loss /= len(test_loader)
    print('\n[Test set] Average loss: {:2.3e}\n'.format(test_loss))
    print('\n[Test set with rigid loss] Average loss: {:2.3e}\n'.format(rigid_loss))
    

    noisy_imgs = noisy_imgs.detach().cpu().view(bshape)
    rigid_nmlz_imgs = rigid_nmlz_imgs.detach().cpu().view(bshape)
    dec_imgs = dec_imgs.detach().cpu().view(bshape)
    
    fig,ax = plt.subplots(3,1,figsize=(10,5))
    grid_im = vutils.make_grid(noisy_imgs, padding=2, normalize=True, nrow=16)
    ax[0].imshow(grid_im.permute(1,2,0))
    grid_im = vutils.make_grid(rigid_nmlz_imgs, padding=2, normalize=False, nrow=16)
    ax[1].imshow(grid_im.permute(1,2,0))
    grid_im = vutils.make_grid(dec_imgs, padding=2, normalize=False, nrow=16)
    ax[2].imshow(grid_im.permute(1,2,0))
    plt.savefig(f"./gan_examples_e{epoch}o{num_epochs}g{cfg.gpuid}.png")

    return test_loss



def run_me():
    args = get_args()
    cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.use_apex = False

    gpuid = 1 # set gpuid
    cfg.N = 2
    cfg.batch_size = int(256 / cfg.N)

    cfg.gpuid = gpuid
    cfg.device = f"cuda:{gpuid}"
    cfg.simcl.device = cfg.device
    cfg.use_collate = True
    cfg.dataset.download = False
    cfg.cls = cfg
    
    torch.cuda.set_device(gpuid)

    # cfg.batch_size = 256
    cfg.init_lr = 5e-5
    print(cfg.batch_size,cfg.init_lr)

    model_gen = load_model_gen(cfg)
    optimizer_gen = load_optimizer_gen(cfg,model_gen)
    scheduler_gen = load_scheduler_gen(cfg,model_gen,optimizer_gen)

    model_disc = load_model_disc(cfg,use_simclr=True)
    optimizer_disc = load_optimizer_disc(cfg,model_disc)
    schduler_disc = load_scheduler_disc(cfg,model_gen,optimizer_disc)

    data,loader = load_dataset(cfg,'denoising')

    model_simclr = load_model_simclr(cfg)
    criterion = nn.BCELoss()

    num_epochs = 100
    cfg.nz = 100
    fixed_noise,_ = next(iter(loader.val))
    p_shape = fixed_noise.shape[2:]
    fixed_noise = fixed_noise.view((cfg.N*cfg.batch_size,)+p_shape).to(cfg.device)


    D_losses = []
    G_losses = []
    img_list = []
    # test_denoising(cfg, model_gen, loader.te,-1,num_epochs)
    for epoch in range(num_epochs):
        losses = train_loop(cfg,model_gen,model_disc,optimizer_gen,optimizer_disc,
                            model_simclr,criterion,loader.tr,epoch,num_epochs,fixed_noise)
        D_losses.extend(losses[0])
        G_losses.extend(losses[1])
        img_list.extend(losses[2])
        denoise_loss = test_denoising(cfg, model_gen, loader.te, epoch, num_epochs)
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
