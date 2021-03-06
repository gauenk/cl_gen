"""
Test Disent Loss function on standard problem
"""

"""
Experiments:

0.) static box on mnist
1.) static noise over all mnist
2.) moving mnist digit

"""

# python imports
import sys,os
sys.path.append("./lib/")
import numpy as np
from easydict import EasyDict as edict
import pathlib
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
import numpy.random as npr

# torch imports
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


# project imports
import settings
from pyutils.cfg import get_cfg
from layers import NT_Xent,SimCLR,get_resnet,LogisticRegression
from layers import DisentangleLoss,Encoder,Decoder,Projector
from learning.train import thtrain_cl as train_cl
from learning.train import thtrain_cls as train_cls
from learning.train import thtrain_disent as train_disent
from learning.test import thtest_cls as test_cls
from learning.utils import save_model,save_optim
from torchvision.datasets import CIFAR10
from datasets import get_cifar10_dataset,get_mnist_dataset

def get_model_epoch_info(cfg):
    if cfg.load:
        return 0,cfg.epoch_num+1
    else: return 0,0


def exploring_nt_xent_loss(cfg):
    print("Exploring the NT_Xent loss function.")

    # load the data
    data,loader = get_mnist_dataset(cfg,'disent')
    img_set,img = next(iter(loader.tr))
    print(len(img_set))
    print(len(img_set[0]))
    print(len(img))
    print(len(img[0]))
    # x = [x_i.to('cpu')[0].permute((1,2,0)).numpy() for x_i in x]
    # x_dec = [x_dec_i.to('cpu')[0].detach().permute((1,2,0)).numpy() for x_dec_i in x_dec]

    batch_size = cfg.disent.batch_size
    loss = NT_Xent(batch_size,0.1,cfg.disent.device,1)
    print(img.reshape(batch_size,-1).shape)
    
    _,img1 = next(iter(loader.tr))
    _,img2 = next(iter(loader.tr))
    img1 = img1.reshape(batch_size,-1)
    img2 = img2.reshape(batch_size,-1)
    print(loss(img1,img2))

    fig,ax = plt.subplots(len(img_set),2)
    for i,img_i in enumerate(img_set):
        # x_recon = x_dec_i - np.min(x_dec_i)
        # x_recon = x_recon / np.max(x_recon)
        # print(x_noisy.shape,x_recon.shape)
        ax[i,0].imshow(img_i[0])
        ax[i,1].imshow(img[0])
    plt.savefig("mnist_rand_blocks.png")
    plt.clf()
    plt.cla()
    

def load_encoder(cfg,enc_type):
    model = Encoder()
    if cfg.disent.load:
        fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
        model_fp = Path(cfg.disent.model_path) / Path(f"enc_{enc_type}") / fn
        model.load_state_dict(torch.load(model_fp, map_location=cfg.disent.device.type))
    model = model.to(cfg.disent.device)
    return model

def load_decoder(cfg):
    model = Decoder() 
    if cfg.disent.load:
        fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
        model_fp = Path(cfg.disent.model_path) / Path("dec") / fn
        model.load_state_dict(torch.load(model_fp, map_location=cfg.disent.device.type))
    model = model.to(cfg.disent.device)
    return model

def load_projector(cfg):
    model = Projector()
    if cfg.disent.load:
        fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
        model_fp = Path(cfg.disent.model_path) / Path("proj") / fn
        model.load_state_dict(torch.load(model_fp, map_location=cfg.disent.device.type))
    model = model.to(cfg.disent.device)
    return model

def get_disent_optim(models):
    params = []
    for name,model in models.items():
        params += list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    if cfg.disent.load:
        fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
        optim_fn = Path(cfg.disent.optim_path) / fn
        optimizer.load_state_dict(torch.load(optim_fn, map_location=cfg.disent.device.type))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience = 5,
                                                           factor=1./np.sqrt(10))
    return optimizer,scheduler

def save_disent_models(cfg,models,optimizer):
    for name,model in models.items():
        saved_mp = cfg.disent.model_path
        cfg.disent.model_path = Path(cfg.disent.model_path) / Path(name)
        save_model(cfg.disent, model, None)
        cfg.disent.model_path = saved_mp
    save_optim(cfg.disent, optimizer)

def load_disent_models(cfg):
    models = edict()
    models.enc_c = load_encoder(cfg,'c')
    models.enc_d = load_encoder(cfg,'d')
    models.dec = load_decoder(cfg)
    models.proj = load_projector(cfg)
    return models

def train_disent_exp(cfg):
    print("Training disentangled representations.")

    # load the data
    data,loader = get_mnist_dataset(cfg,'disent')

    # load the model and set criterion
    models = load_disent_models(cfg)
    hyperparams = edict()
    hyperparams.h = 0.5
    hyperparams.g = 0.5
    hyperparams.x = 0.5
    hyperparams.temperature = 0.1
    criterion = DisentangleLoss(models,hyperparams,
                                cfg.disent.N,
                                cfg.disent.batch_size,
                                cfg.disent.device)
    optimizer,scheduler = get_disent_optim(models)

    # init the training loop
    writer = SummaryWriter()
    global_step,current_epoch = get_model_epoch_info(cfg.disent)
    cfg.disent.global_step = global_step
    cfg.disent.current_epoch = current_epoch

    # training loop
    for epoch in range(cfg.disent.current_epoch, cfg.disent.epochs+1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train_disent(cfg.disent, loader.tr, models,
                                  criterion, optimizer, epoch, writer)
        if scheduler:
           scheduler.step(loss_epoch)

        if epoch % cfg.disent.checkpoint_interval == 0:
            save_disent_models(cfg,models,optimizer)

        writer.add_scalar("Loss/train", loss_epoch / len(loader.tr), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{cfg.disent.epochs}]\t Loss: {loss_epoch / len(loader.tr)}\t lr: {round(lr, 5)}"
        )
        cfg.disent.current_epoch += 1

    save_disent_models(cfg,models,optimizer)

def test_disent(cfg):
    print("Testing image disentanglement.")

    # load the data
    data,loader = get_mnist_dataset(cfg,'disent')

    # load the model and set criterion
    models = load_disent_models(cfg)
    for name,model in models.items(): model.eval()

    # get the data
    x = next(iter(loader.tr))[0]
    pics = [x_i.to(cfg.disent.device) for x_i in x]
    nt = len(pics) 
    fig,ax = plt.subplots(nt,nt+1,figsize=(8,8))
    for i,pic_i in enumerate(pics):
        plot_th_tensor(ax,i,-1,pic_i)
        encC_i = models.enc_c(pic_i)
        encD_i = models.enc_d(pic_i)
        for j,pic_j in enumerate(pics):
            encC_j = models.enc_c(pic_j)
            encD_j = models.enc_d(pic_j)
            dec_ij = models.dec(torch.cat([encC_i,encD_j],dim=1))
            plot_th_tensor(ax,i,j,dec_ij)
            if i == j: continue
            dec_ji = models.dec(torch.cat([encC_j,encD_i],dim=1))
            # plot_th_tensor(ax,j,i,dec_ji)
    plt.savefig(f"test_disentangle_{cfg.disent.epoch_num}.png")
    plt.clf()
    plt.cla()

def test_disent_over_epochs(cfg):
    epoch_num_list = [0,10,50,100,120,150]
    for epoch_num in epoch_num_list:
        cfg.disent.epoch_num = epoch_num
        test_disent(cfg)        

def plot_th_tensor(ax,i,j,dec_ij):
    dec_ij = dec_ij.to('cpu').detach().numpy()[0,0]
    ax[i,j].imshow(dec_ij,  cmap='Greys_r',  interpolation=None)


if __name__ == "__main__":
    cfg = get_cfg()
    cfg.exp_name = "static_noise"

    cfg.disent = edict()
    cfg.disent.epochs = 5000
    cfg.disent.load = False
    cfg.disent.epoch_num = 0

    model_path = Path(f"{settings.ROOT_PATH}/output/disent/{cfg.exp_name}/")
    optim_path = Path(f"{settings.ROOT_PATH}/output/disent/{cfg.exp_name}/optim/")
    if not model_path.exists(): model_path.mkdir(parents=True)
    cfg.disent.model_path = model_path
    cfg.disent.optim_path = optim_path
    
    cfg.disent.workers = 1
    cfg.disent.batch_size = 128
    cfg.disent.global_step = 0
    cfg.disent.device = cfg.cl.device
    cfg.disent.current_epoch = 0
    cfg.disent.checkpoint_interval = 1
    cfg.disent.log_interval = 1


    cfg.disent.dataset = edict()
    cfg.disent.dataset.name = 'MNIST'
    cfg.disent.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.disent.dataset.n_classes = 10
    cfg.disent.dataset.noise_levels = [5e-2,5e-2]
    cfg.disent.N = 5

    # exploring_nt_xent_loss(cfg)
    train_disent_exp(cfg)

    cfg.disent.load = True
    cfg.disent.epoch_num = 175
    cfg.disent.batch_size = 1
    # test_disent_over_epochs(cfg)
    # test_disent(cfg)
