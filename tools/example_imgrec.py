"""
Test ImgRec Loss function on standard problem
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
from layers import ImgRecLoss,ImageDecoder,load_simclr
from learning.train import thtrain_cl as train_cl
from learning.train import thtrain_cls as train_cls
from learning.train import thtrain_imgrec as train_imgrec
from learning.test import thtest_cls as test_cls
from learning.utils import save_model
from torchvision.datasets import CIFAR10
from datasets import get_cifar10_dataset

def get_model_epoch_info(cfg):
    if cfg.load:
        return 0,cfg.epoch_num+1
    else: return 0,0

def get_contrastive_learning_models(cfg):
    model,_,_ = load_simclr(cfg)
    encoder,projector = model.encoder,model.projector
    encoder.eval()
    projector.eval()
    return encoder,projector

def get_image_reconstruction_model(cfg):
    model = ImageDecoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=1./np.sqrt(10))

    if cfg.imgrec.load:
        fn = Path("checkpoint_{}.tar".format(cfg.imgrec.epoch_num))
        model_fp = Path(cfg.imgrec.model_path) / fn
        model.load_state_dict(torch.load(model_fp, map_location=cfg.imgrec.device.type))
    model = model.to(cfg.imgrec.device)

    return model,optimizer,scheduler

def train_nonblind_imgrec(cfg):
    print("Training non-blind image reconstruction.")

    # load the data
    data,loader = get_cifar10_dataset(cfg,'imgrec')
    transforms = data.transforms

    # load the model and set criterion
    encoder,projection = get_contrastive_learning_models(cfg)
    decoder,optimizer,scheduler = get_image_reconstruction_model(cfg)
    criterion = ImgRecLoss(decoder, encoder, projection, blind = False,
                           gamma = 0.1, beta = 0.1)

    # init the training loop
    writer = SummaryWriter()
    global_step,current_epoch = get_model_epoch_info(cfg.imgrec)
    cfg.imgrec.global_step = global_step
    cfg.imgrec.current_epoch = current_epoch

    # training loop
    for epoch in range(cfg.imgrec.current_epoch, cfg.imgrec.epochs+1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train_imgrec(cfg.imgrec, loader.tr, transforms.tr,
                                  decoder, criterion, optimizer,
                                  epoch, writer)

        #if scheduler:
        #    scheduler.step()

        if epoch % cfg.imgrec.checkpoint_interval == 0:
            save_model(cfg.imgrec, decoder, optimizer)

        writer.add_scalar("Loss/train", loss_epoch / len(loader.tr), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{cfg.imgrec.epochs}]\t Loss: {loss_epoch / len(loader.tr)}\t lr: {round(lr, 5)}"
        )
        cfg.imgrec.current_epoch += 1

    save_model(cfg.imgrec, decoder, optimizer)

def test_nonblind_imgrec(cfg):
    print("Testing non-blind image reconstruction.")

    # load the data
    data,loader = get_cifar10_dataset(cfg,'imgrec')
    transforms = data.transforms

    # load the model and set criterion
    encoder,projection = get_contrastive_learning_models(cfg)
    decoder,optimizer,scheduler = get_image_reconstruction_model(cfg)
    criterion = ImgRecLoss(decoder, encoder, projection, blind = False,
                           gamma = 0.1, beta = 0.1)
    encoder.eval()
    projection.eval()
    decoder.eval()

    # get the data
    # x = data.tr[0][0]
    x = next(iter(loader.tr))[0]
    x = [x_i.to(cfg.imgrec.device) for x_i in x]

    x_enc = [encoder(x_i) for x_i in x]
    x_dec = decoder(x_enc)
    print(x_dec[0].shape)

    x = [x_i.to('cpu')[0].permute((1,2,0)).numpy() for x_i in x]
    x_dec = [x_dec_i.to('cpu')[0].detach().permute((1,2,0)).numpy() for x_dec_i in x_dec]
    fig,ax = plt.subplots(2,len(x_dec))
    for i,x_dec_i in enumerate(x_dec):
        print(x[i].shape,x_dec_i.shape)
        x_noisy = x[i]
        x_recon = x_dec_i - np.min(x_dec_i)
        x_recon = x_recon / np.max(x_recon)
        print(x_noisy.shape,x_recon.shape)
        ax[i,0].imshow(x_noisy)
        ax[i,1].imshow(x_recon)
    plt.savefig("test_nonblind_imgrec.png")
    plt.clf()
    plt.cla()

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.exp_name = "imgrec_bw"
    cfg.cl.load = True
    cfg.cl.epoch_num = 1000

    cfg.imgrec = edict()

    cfg.imgrec.epochs = 100
    cfg.imgrec.load = False
    cfg.imgrec.epoch_num = 0

    model_path = Path(f"{settings.ROOT_PATH}/output/imgrec/{cfg.exp_name}/{cfg.cl.epoch_num}/")
    if not model_path.exists(): model_path.mkdir(parents=True)
    cfg.imgrec.model_path = model_path
    
    cfg.imgrec.workers = 1
    cfg.imgrec.batch_size = 50
    cfg.imgrec.global_step = 0
    cfg.imgrec.device = cfg.cl.device
    cfg.imgrec.current_epoch = 0
    cfg.imgrec.checkpoint_interval = 10
    cfg.imgrec.log_interval = cfg.cl.log_interval

    cfg.imgrec.dataset = edict()
    cfg.imgrec.dataset.name = 'CIFAR10'
    cfg.imgrec.dataset.root = "/home/gauenk/data/cifar10/"
    cfg.imgrec.dataset.n_classes = 10
    cfg.imgrec.dataset.noise_levels = [5e-2,5e-2]

    # train_nonblind_imgrec(cfg)

    cfg.imgrec.load = True
    cfg.imgrec.epoch_num = 40
    cfg.imgrec.batch_size = 1
    test_nonblind_imgrec(cfg)
