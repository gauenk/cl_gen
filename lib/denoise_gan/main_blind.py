
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
from layers.simcl import ClBlockLoss

# [this folder] project code
from denoise_gan.config import get_cfg,get_args
from .model_io import load_model_gen,load_model_disc,load_model_simclr,load_model_rec,load_model_noise
from .optim_io import load_optimizer_gen,load_optimizer_disc
from .sched_io import load_scheduler_gen,load_scheduler_disc

def get_disc_update_bool(batch_idx,epoch):
    epoch_0 = epoch == 0
    batch_g0 = batch_idx > 0
    batch_3 = (batch_idx % 3) == 0 and batch_g0
    batch_5 = (batch_idx % 5) == 0 and batch_g0
    batch_10 = (batch_idx % 10) == 0 and batch_g0
    epoch_3 = epoch == 3

    cond = epoch_0 and batch_3
    cond = cond or (batch_5)

    return True

def get_noisy_update_bool(batch_idx,epoch):
    epoch_0 = epoch == 0
    epoch_g0 = epoch > 0
    batch_g0 = batch_idx > 0
    batch_3 = (batch_idx % 3) == 0 and batch_g0
    batch_5 = (batch_idx % 5) == 0 and batch_g0
    batch_10 = (batch_idx % 10) == 0 and batch_g0
    batch_100 = (batch_idx % 100) == 0 and batch_g0
    epoch_3 = epoch == 3

    cond = epoch_0 and batch_100
    cond = cond or (epoch_g0 and batch_5)
    return cond

def get_rec_update_bool(batch_idx,epoch):
    epoch_0 = epoch == 0
    epoch_g0 = epoch > 0
    batch_g0 = batch_idx > 0
    batch_3 = (batch_idx % 3) == 0 and batch_g0
    batch_5 = (batch_idx % 5) == 0 and batch_g0
    batch_10 = (batch_idx % 10) == 0 and batch_g0
    batch_100 = (batch_idx % 100) == 0 and batch_g0
    epoch_3 = epoch == 3

    cond = epoch_0 and batch_100
    cond = cond or (epoch_g0 and batch_5)
    return cond


def train_loop(cfg,model_disc,model_rec,model_noise,
               optimizer_noise,optimizer_rec,optimizer_disc,
               model_simclr,criterion,train_loader,
               epoch,num_epochs,fixed_noise):

    device = cfg.device
    real_label = 1 #cfg.real_label
    fake_label = not real_label
    nz = 100 # cfg.nz
    cfg.log_interval = 3
    cfg.train_gen_interval = 500
    # need: cfg.log_interval,cfg.train_gen_interval
    simclr_loss = ClBlockLoss(cfg.hyper_params,2*cfg.N,cfg.batch_size)
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
        noisy_imgs = noisy_imgs.view(p_view)
        disc_update = get_disc_update_bool(batch_idx,epoch)

        # -- (i) real images
        noisy_input = noisy_imgs
        if cfg.use_simclr == "all":
            noisy_emb,_ = model_simclr(noisy_imgs)
            noisy_emb = noisy_emb.view(N*BS,2,32,32)
            noisy_input = noisy_emb.detach()
        print(noisy_input.shape)
        output = model_disc(noisy_input).view(-1)
        err_disc_real = output.mean(0).view(1)
        if disc_update:
            err_disc_real.backward(one)
        D_x = output.mean().item()

        # -- (ii) fake images
        fake = model_rec(noisy_imgs).view(p_view)
        fake_noisy = model_noise(fake,fake-noisy_imgs)
        fake_input = fake_noisy
        if cfg.use_simclr == "all":
            fake_emb,_ = model_simclr(fake_noisy.view(n_view))
            fake_emb = fake_emb.view(N*BS,2,32,32)
            fake_input = fake_emb.detach()
        output = model_disc(fake_input).view(-1)
        err_disc_fake = output.mean(0).view(1)
        if disc_update:
            grad_penalty = calc_gradient_penalty(cfg, model_disc, noisy_input, fake_input)
            grad_penalty.backward()
            err_disc_fake.backward(mone)
        D_G_z1 = output.mean().item()
        error_disc = err_disc_real - err_disc_fake
        if disc_update:
            optimizer_disc.step()
        
        # for p in model_disc.parameters():
        #     p.data.clamp_(-0.01, 0.01)
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

        model_rec.zero_grad()
        model_noise.zero_grad()


        noisy_update = get_noisy_update_bool(batch_idx,epoch)
        rec_update = get_rec_update_bool(batch_idx,epoch)
        rec_update = rec_update or noisy_update

        gen_loss = 0
        if rec_update:
            # include reconstruction loss.
            if cfg.use_simclr != "none" and cfg.use_simclr != False:
                if cfg.use_simclr != "all":
                    noisy_emb,_ = model_simclr(noisy_imgs.view(n_view))
                    noisy_emb = noisy_emb.view(N*BS,2,32,32)
                    fake_emb,_ = model_simclr(fake_noisy.view(n_view))
                    fake_emb = fake_emb.view(N*BS,2,32,32)
                    offset_idx = [(i+1)%N for i in range(N)]
                    noisy_emb = noisy_emb.view((N,BS,2,32,32))
                    # fake_emb = fake_emb.view(N,BS,2,32,32)[offset_idx]
                fake_emb = fake_emb.view(N,BS,2,32,32)
                # embs = torch.stack([fake_emb,noisy_emb])
                embs = torch.cat([fake_emb,noisy_emb],dim=0)
                rec_loss = simclr_loss(embs)
            else:
                offset_idx = [(i+1)%N for i in range(N)]
                noisy_imgs = noisy_imgs.view(dshape)
                fake = fake.view(dshape)
                offset_idx = [(i+1)%N for i in range(N)]
                fake_offset = fake[offset_idx]
                rec_loss = F.mse_loss(fake_offset,noisy_imgs)
            gen_loss = rec_loss.view(1)

        if noisy_update:
        # if ((batch_idx % 5) == 0 and batch_idx > 0 and epoch > 10) or ((batch_idx % 10) == 0 and batch_idx > 0):
            if cfg.use_simclr == "all":
                output = model_disc(fake_emb).view(-1)
            else:
                output = model_disc(fake_noisy).view(-1)
            error_gen = output.mean(0).view(1)
            gen_loss += error_gen
            # error_gen.backward(one,retain_graph=True)
            D_G_z2 = output.mean().item()

        if rec_update or noisy_update: gen_loss.backward()
        if noisy_update: optimizer_noise.step()
        if rec_update: optimizer_rec.step()

        if (batch_idx % cfg.log_interval) == 0:
            print("[%d/%d][%d/%d] Loss_D: %.4f\tLoss_G %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"%
                  (epoch, num_epochs, batch_idx, len(train_loader),error_disc.item(),
                   error_gen.item(),D_x,D_G_z1,D_G_z2))

        D_losses.append(error_disc)
        G_losses.append(error_gen)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (batch_idx % cfg.train_gen_interval == 0) or ((epoch == num_epochs-1) and (batch_idx == len(train_loader)-1)):
            with torch.no_grad():
                fake = model_rec(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=16))

    return D_losses,G_losses,img_list
        
def calc_gradient_penalty(cfg, netD, real_data, fake_data):
    #print real_data.size()
    BS = real_data.shape[0]
    alpha = torch.rand(BS, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    use_cuda = True

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu)
                              if use_cuda else torch.ones(
                                      disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * cfg.grad_lambda
    return gradient_penalty

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

    gpuid = 0 # set gpuid
    cfg.N = 2
    cfg.batch_size = int(256 / cfg.N)
    cfg.use_simclr = "some"
    cfg.hyper_params.temperature = 0.5


    cfg.gpuid = gpuid
    cfg.device = f"cuda:{gpuid}"
    cfg.simcl.device = cfg.device
    cfg.use_collate = True
    cfg.dataset.download = False
    cfg.cls = cfg
    cfg.grad_lambda = 50
    
    torch.cuda.set_device(gpuid)

    # cfg.batch_size = 256
    cfg.init_lr = 5e-5
    print(cfg.batch_size,cfg.init_lr)

    model_rec = load_model_rec(cfg)
    optimizer_rec = load_optimizer_gen(cfg,model_rec)
    scheduler_rec = load_scheduler_gen(cfg,model_rec,optimizer_rec)

    model_noise = load_model_noise(cfg)
    optimizer_noise = load_optimizer_gen(cfg,model_noise)
    scheduler_noise = load_scheduler_gen(cfg,model_noise,optimizer_noise)

    bool_simclr = (cfg.use_simclr == "all")
    model_disc = load_model_disc(cfg,use_simclr=bool_simclr)
    optimizer_disc = load_optimizer_disc(cfg,model_disc)
    schduler_disc = load_scheduler_disc(cfg,model_disc,optimizer_disc)

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
    # test_denoising(cfg, model_rec, loader.te,-1,num_epochs)
    for epoch in range(num_epochs):
        losses = train_loop(cfg,model_disc,model_rec,model_noise,
                            optimizer_noise,optimizer_rec,optimizer_disc,
                            model_simclr,criterion,loader.tr,
                            epoch,num_epochs,fixed_noise)
        D_losses.extend(losses[0])
        G_losses.extend(losses[1])
        img_list.extend(losses[2])
        denoise_loss = test_denoising(cfg, model_rec, loader.te, epoch, num_epochs)
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
