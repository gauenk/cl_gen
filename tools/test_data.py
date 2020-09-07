# python imports
import sys,os,json,argparse
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
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


# project imports
import settings
from pyutils.cfg import get_cfg
from pyutils.misc import np_log
from layers import NT_Xent,SimCLR,get_resnet,LogisticRegression,DisentangleStaticNoiseLoss
from layers import DisentangleLoss,Encoder,Decoder,Projector
from learning.train import thtrain_cl as train_cl
from learning.train import thtrain_cls as train_cls
from learning.train import thtrain_disent as train_disent
from learning.test import thtest_cls as test_cls
from learning.test import thtest_static as test_static
from learning.utils import save_model,save_optim
from torchvision.datasets import CIFAR10
from datasets import get_dataset


def get_args():
    parser = argparse.ArgumentParser(description="Run static denoising on test data")
    parser.add_argument("--noise-level", type=float, default=1e-2,
                        help="noise level for each input image")
    parser.add_argument("--N", type=int, default=2,
                        help="number of noisy images to generate")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="how big are the batch sizes")
    parser.add_argument("--dataset", type=str, default="MNIST",
                        help="which dataset to create")
    parser.add_argument("--loss-type", type=str, default="simclr",
                        help="which loss function to use")
    args = parser.parse_args()
    return args

def get_denoising_cfg(args):
    cfg = get_cfg()

    cfg.cl.device = torch.device("cuda:{}".format(0))
    cfg.cls.device = cfg.cl.device


    # set the name
    cfg.exp_name = "test_data"

    cfg.disent = edict()
    cfg.disent.epochs = 10

    cfg.disent.load = False
    cfg.disent.epoch_num = 0

    cfg.disent.dataset = edict()
    cfg.disent.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.disent.dataset.n_classes = 10
    cfg.disent.dataset.name = args.dataset
    cfg.disent.noise_level = args.noise_level
    cfg.disent.N = args.N
    cfg.disent.loss_type = args.loss_type


    dsname = cfg.disent.dataset.name.lower()
    model_path = Path(f"{settings.ROOT_PATH}/output/disent/{dsname}/{cfg.exp_name}")
    optim_path = Path(f"{settings.ROOT_PATH}/output/disent/{dsname}/{cfg.exp_name}/optim/")
    if not model_path.exists(): model_path.mkdir(parents=True)
    cfg.disent.model_path = model_path
    cfg.disent.optim_path = optim_path
    
    cfg.disent.workers = 1
    cfg.disent.batch_size = args.batch_size
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

    return cfg

def plot_th_tensor(ax,i,j,img,title):
    img = torch.clamp(img,-.5,.5)
    img = img.to('cpu').detach().numpy()[0].transpose(1,2,0)
    img += 0.5
    ax[i,j].imshow(img,  interpolation=None)
    ax[i,j].set_xticks([])
    ax[i,j].set_yticks([])
    ax[i,j].set_title(title)
    
    
def get_test_data_dir(cfg):
    return Path(f"{settings.ROOT_PATH}/test/")

def main():

    args = get_args()
    cfg = get_denoising_cfg(args)
    cfg.disent.random_crop = False
    data,loader = get_dataset(cfg,'disent')
    criterion = nn.MSELoss()
    
    # get the data
    numOfExamples = 4
    fig,ax = plt.subplots(numOfExamples,2,figsize=(8,8))
    for num_ex in range(numOfExamples):
        x,raw_img = next(iter(loader.val))
        pic_i = x[0]
        pic_title = 'noisy'
        if not cfg.disent.random_crop:
            mse = criterion(pic_i,raw_img).item()
            psnr = 10 * np_log(1./mse)[0]/np_log(10)[0]
            pic_title = 'psnr: {:2.2f}'.format(psnr)
        plot_th_tensor(ax,num_ex,0,pic_i,pic_title)
        plot_th_tensor(ax,num_ex,1,raw_img,'raw')
    t_dir = get_test_data_dir(cfg)
    dsname = cfg.disent.dataset.name.lower()
    fn = Path(f"test_dataset_{dsname}.png")
    path = t_dir / fn
    print(f"Writing images to output {path}")
    plt.savefig(path)
    plt.clf()
    plt.cla()


if __name__ == "__main__":
    main()
