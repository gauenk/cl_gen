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
import sys,os,json
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
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import torch.nn.functional as F

# project imports
import settings
from pyutils.cfg import get_cfg
from pyutils.misc import np_log,rescale_noisy_image,get_model_epoch_info
from layers import NT_Xent,SimCLR,get_resnet,LogisticRegression,DisentangleStaticNoiseLoss,Projector
from layers.denoising import DenoisingLoss,Encoder,Decoder,Projector
from layers.denoising import reconstruct_set
from learning.train import thtrain_cl as train_cl
from learning.train import thtrain_cls as train_cls
from learning.train import thtrain_disent as train_disent
from learning.test import thtest_cls as test_cls
from learning.test import thtest_static as test_static
from learning.utils import save_model,save_optim
from torchvision.datasets import CIFAR10
from datasets import get_dataset
from schedulers import get_simclr_scheduler,LinearWarmup,get_train_scheduler

    
def exploring_nt_xent_loss(cfg):
    print("Exploring the NT_Xent loss function.")

    # load the data
    data,loader = get_dataset(cfg,'disent')
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
    nc = cfg.disent.n_channels
    model = Encoder(n_channels=nc,embedding_size = 256)
    # model = nn.DataParallel(model)
    print('encoder',cfg.disent.device.type)
    if cfg.disent.load:
        fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
        model_fp = Path(cfg.disent.model_path) / Path(f"enc_{enc_type}") / fn
        model.load_state_dict(torch.load(model_fp, map_location=cfg.disent.device.type))
    model = model.to(cfg.disent.device)
    return model

def load_decoder(cfg):
    nc = cfg.disent.n_channels
    model = Decoder(n_channels=nc,embedding_size=256)
    # model = nn.DataParallel(model)
    print('decoder',cfg.disent.device.type)
    if cfg.disent.load:
        fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
        model_fp = Path(cfg.disent.model_path) / Path("dec") / fn
        model.load_state_dict(torch.load(model_fp, map_location=cfg.disent.device.type))
    model = model.to(cfg.disent.device)
    return model

def load_projector(cfg):
    model = Projector(n_features=768)
    if cfg.disent.load:
        fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
        model_fp = Path(cfg.disent.model_path) / Path("proj") / fn
        model.load_state_dict(torch.load(model_fp, map_location=cfg.disent.device.type))
    model = model.to(cfg.disent.device)
    return model

def get_disent_optim(cfg,models,tr_batches=None):
    params = []
    for name,model in models.items():
        params += list(model.parameters())
    base_lr = 1e-3
    lr = base_lr * 1 # cfg.disent.batch_size
    optimizer = torch.optim.Adam(params, lr=lr)
    # optimizer = torch.optim.SGD(params, lr=1e-3)
    # optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, dampening=0, nesterov=True)
    # if cfg.disent.load:
    #     fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
    #     optim_fn = Path(cfg.disent.optim_path) / fn
    #     optimizer.load_state_dict(torch.load(optim_fn, map_location=cfg.disent.device.type))
    # milestones = [400]
    # milestones = [75,150,350]
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-3,
    #                                                 steps_per_epoch=tr_batches,
    #                                                 epochs=cfg.disent.epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience = 10,
                                                           factor=1./np.sqrt(10))
    batch_size = cfg.disent.batch_size
    epochs = cfg.disent.epochs
    burnin = 10 # cfg.disent.burnin
    batches_per_epoch = tr_batches
    load_epoch = -1 # cfg.disent.epoch_num
    # scheduler = get_simclr_scheduler(optimizer,batch_size,epochs,burnin,batches_per_epoch,load_epoch)
    return optimizer,scheduler


def save_disent_models(cfg,models,optimizer):
    for name,model in models.items():
        saved_mp = cfg.disent.model_path
        cfg.disent.model_path = Path(cfg.disent.model_path) / Path(name)
        save_model(cfg.disent, model, None)
        cfg.disent.model_path = saved_mp
    save_optim(cfg.disent, optimizer)

def load_static_models(cfg):
    models = edict()
    models.encoder = load_encoder(cfg,'c')
    models.decoder = load_decoder(cfg)
    models.projector = load_projector(cfg)

    # backward compat
    models.enc_c = models.encoder
    models.dec = models.decoder

    return models


def train_disent_exp(cfg):
    print("Training disentangled representations.")
    print("TODO: allow users to load exp A and start new exp B")

    # load the data
    data,loader = get_dataset(cfg,'disent')

    # load the model and set criterion
    models = load_static_models(cfg)
    hyperparams = edict()
    hyperparams.g = 0
    hyperparams.h = cfg.disent.hyper_h
    hyperparams.x = 0
    hyperparams.temperature = 0.1
    criterion = DenoisingLoss(models,hyperparams,
                              cfg.disent.N,
                              cfg.disent.batch_size,
                              cfg.disent.device,
                              cfg.disent.img_loss_type,
                              'simclr',
                              cfg.disent.agg_enc_fxn)
    optimizer,scheduler = get_disent_optim(cfg,models,len(loader.tr))

    # init writer
    writer = SummaryWriter(filename_suffix=cfg.exp_name)

    # test init model
    # if cfg.disent.load is False:
    #     te_loss = test_static(cfg.disent,models.enc_c,models.dec,loader.te)            
    #     cfg.disent.current_epoch = -1
    #     writer.add_scalar("Loss/test", te_loss, -1)
    #     save_disent_models(cfg,models,optimizer)

    # init training loop
    global_step,current_epoch = get_model_epoch_info(cfg.disent)
    cfg.disent.global_step = global_step
    cfg.disent.current_epoch = current_epoch

    # training loop
    tr_scheduler = get_train_scheduler(scheduler)
    test_losses = {}
    for epoch in range(cfg.disent.current_epoch, cfg.disent.epochs+1):
        lr = optimizer.param_groups[0]["lr"]

        loss_epoch = train_disent(cfg.disent, loader.tr, models,
                                  criterion, optimizer, epoch, writer,
                                  tr_scheduler)

        if scheduler and tr_scheduler is None:
            val_loss = test_static(cfg.disent,models.enc_c,models.dec,loader.val)
            scheduler.step(val_loss)

        if epoch % cfg.disent.checkpoint_interval == 0:
            save_disent_models(cfg,models,optimizer)

        if epoch % cfg.disent.test_interval == 0:
            te_loss = test_static(cfg.disent,models.enc_c,models.dec,loader.te)
            writer.add_scalar("Loss/test", te_loss, epoch)

        writer.add_scalar("Loss/train", loss_epoch / len(loader.tr), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{cfg.disent.epochs}]\t Loss: {loss_epoch / len(loader.tr)}\t " + "{:2.3e}".format(lr)
        )
        cfg.disent.current_epoch += 1

    save_disent_models(cfg,models,optimizer)

def plot_noise_floor(cfg):
    """
    Plot the noise level for Gaussian random noise
    """
    mean = 1.290e-02
    stddev = 1.149e-02
    val = mean
    # val = mean + stddev
    val = mean - stddev
    writer = SummaryWriter(filename_suffix=cfg.exp_name)
    for epoch in range(cfg.disent.epochs+1):
        writer.add_scalar(f"test_loss at epoch", val, epoch)
        writer.add_scalar(f"Loss/train", val, epoch)

def test_disent(cfg,n_runs=1,use_psnr=False):
    print(f"Testing image denoising with epoch {cfg.disent.epoch_num}")

    # load the data
    data,loader = get_dataset(cfg,'disent')

    # load the model and set criterion
    models = load_static_models(cfg)
    for name,model in models.items(): model.eval()

    # tr_loss = test_static(cfg.disent,models.enc_c,models.dec,loader.tr)
    # val_loss = test_static(cfg.disent,models.enc_c,models.dec,loader.val)
    te_losses = []
    for n in range(n_runs):
        te_loss = test_static(cfg.disent,models.enc_c,
                              models.dec,loader.te,use_psnr=use_psnr)
        te_losses.append(te_loss)
    if n_runs > 1:
        mean = np.mean(te_losses)
        stderr = np.std(te_losses) / np.sqrt(len(te_losses))
    else:
        mean = te_losses[0]
        stderr = 0.
    # print("Testing loss: {:.3f}".format(tr_loss))
    # print("Testing loss: {:.3f}".format(val_loss))
    print("Testing loss: {:2.3e} +/- {:2.3e}".format(mean,1.96*stderr))
    losses = edict()
    losses.te_losses = te_losses
    losses.mean = mean
    losses.stderr = stderr
    return losses

def test_disent_examples(cfg):
    print(f"Testing image denoising with epoch {cfg.disent.epoch_num}")

    # load the data
    data,loader = get_dataset(cfg,'disent')

    # load the model and set criterion
    models = load_static_models(cfg)
    for name,model in models.items(): model.eval()

    # get the data
    enc,dec = models.enc_c,models.dec
    numOfExamples = 4
    fig,ax = plt.subplots(numOfExamples,3,figsize=(8,8))
    for num_ex in range(numOfExamples):
        pic_set,raw_img = next(iter(loader.te))
        pic_set = pic_set.to(cfg.disent.device)
        raw_img = raw_img.to(cfg.disent.device)

        N = len(pic_set)
        BS = len(pic_set[0])
        pshape = pic_set[0][0].shape
        shape = (N,BS,) + pshape

        rec_set = reconstruct_set(pic_set,enc,dec,cfg.disent.share_enc)
        rec_set = rescale_noisy_image(rec_set)
        rec_set_i = rec_set[0]

        # Plot Decoded Image
        mse = F.mse_loss(rec_set_i,raw_img).item()
        psnr = 10 * np_log(1./mse)[0]/np_log(10)[0]
        pic_title = 'rec psnr: {:2.2f}'.format(psnr)
        plot_th_tensor(ax,num_ex,0,rec_set_i,pic_title)

        # Plot Noisy Image
        pic_i = rescale_noisy_image(pic_set[0])
        mse = F.mse_loss(pic_i,raw_img).item()
        psnr = 10 * np_log(1./mse)[0]/np_log(10)[0]
        pic_title = 'noisy psnr: {:2.2f}'.format(psnr)
        plot_th_tensor(ax,num_ex,1,pic_i,pic_title)

        # Plot Clean Image
        pic_title = 'raw'
        plot_th_tensor(ax,num_ex,2,raw_img,pic_title)

    exp_report_dir = get_report_dir(cfg)
    fn = Path(f"test_disentangle_{cfg.exp_name}_{cfg.disent.epoch_num}.png")
    path = exp_report_dir / fn
    print(f"Writing images to output {path}")
    plt.savefig(path)
    plt.clf()
    plt.cla()

def get_report_dir(cfg):
    base = Path(f"{settings.ROOT_PATH}/reports/")
    base = base / f"noise_{cfg.disent.noise_level}"
    base = base / f"N_{cfg.disent.N}"
    base = base / f"name_{cfg.exp_name}"
    if not base.exists(): base.mkdir(parents=True)
    return base

def test_disent_over_epochs(cfg,epoch_num_list):
    means,stderrs = {},{}
    for epoch_num in epoch_num_list:
        cfg.disent.epoch_num = epoch_num
        losses = test_disent(cfg)
        mean,stderr = losses.mean,losses.stderr
        means[epoch_num] = mean 
        stderrs[epoch_num] = stderr
    print("Losses by epoch")
    print("means")
    print(means)
    print("stderrs")
    print(stderrs)
    losses = {'means':means,'stderrs':stderrs}
    write_losses(cfg,losses)

def write_losses(cfg,losses):
    fn = get_unique_write_losses_fn(cfg,losses)
    with open(fn,'w+') as f:
        f.write(json.dumps(losses))

def get_unique_write_losses_fn(cfg,losses):
    exp_report_dir = get_report_dir(cfg)
    fn = Path(f"{settings.ROOT_PATH}/reports/{cfg.exp_name}.txt")
    path = exp_report_dir / fn
    num = 2
    while path.exists():
        fn = Path(f"{settings.ROOT_PATH}/reports/{cfg.exp_name}_{num}.txt")
        path = exp_report_dir / fn
        num += 1
    return fn
    

def test_disent_examples_over_epochs(cfg,epoch_num_list):
    for epoch_num in epoch_num_list:
        cfg.disent.epoch_num = epoch_num
        test_disent_examples(cfg)        

def plot_th_tensor(ax,i,j,dec_ij,title):
    dec_ij = dec_ij.to('cpu').detach().numpy()[0,0]
    dec_ij += np.abs(np.min(dec_ij))
    dec_ij = dec_ij / dec_ij.max()
    ax[i,j].imshow(dec_ij,  cmap='Greys_r',  interpolation=None)
    ax[i,j].set_xticks([])
    ax[i,j].set_yticks([])
    ax[i,j].set_title(title)

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.exp_name = "static_noise_cifar10"
    # cfg.exp_name = "static_noise_celeba"

    cfg.disent = edict()
    cfg.disent.epochs = 200
    cfg.disent.load = False
    cfg.disent.epoch_num = 40

    cfg.disent.dataset = edict()
    cfg.disent.dataset.name = 'CIFAR10'
    # cfg.disent.dataset.name = "celeba"
    cfg.disent.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.disent.dataset.n_classes = 10
    cfg.disent.noise_level = 5e-2
    cfg.disent.N = 5
    cfg.disent.img_loss_type = 'l2' 
    cfg.disent.share_enc = False
    cfg.disent.hyper_h = 0

    dsname = cfg.disent.dataset.name.lower()
    model_path = Path(f"{settings.ROOT_PATH}/output/disent/{cfg.exp_name}/{dsname}")
    optim_path = Path(f"{settings.ROOT_PATH}/output/disent/{cfg.exp_name}/{dsname}/optim/")
    if not model_path.exists(): model_path.mkdir(parents=True)
    cfg.disent.model_path = model_path
    cfg.disent.optim_path = optim_path
    
    cfg.disent.workers = 1
    cfg.disent.batch_size = 128
    cfg.disent.global_step = 0
    cfg.disent.device = cfg.cl.device
    cfg.disent.current_epoch = 0
    cfg.disent.checkpoint_interval = 1
    cfg.disent.test_interval = 5
    cfg.disent.log_interval = 1

    if cfg.disent.dataset.name.lower() == "mnist":
        cfg.disent.n_channels = 1
    else:
        cfg.disent.n_channels = 3

    # exploring_nt_xent_loss(cfg)
    cfg.disent.agg_enc_fxn = 'mean'
    if cfg.disent.agg_fxn is False:
        cfg.disent.agg_enc_fxn = 'id'
    train_disent_exp(cfg)


    cfg.disent.load = True
    cfg.disent.epoch_num = 0
    cfg.disent.batch_size = 20
    cfg.disent.noise_level = 5e-2
    cfg.disent.N = 5
    # test_disent(cfg)
    epoch_num_list = [0,5,50,100,250,450]
    # test_disent_over_epochs(cfg,epoch_num_list)
    # test_disent_examples(cfg)
    # epoch_num_list = [0,5,50,100,250,450]
    # test_disent_examples_over_epochs(cfg,epoch_num_list)
