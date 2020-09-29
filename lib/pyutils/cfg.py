# python imports
from easydict import EasyDict as edict
from pathlib import Path

# pytorch imports
import torch

# project imports
from settings import ROOT_PATH


#
# Old "get_cfg" function
# 

def get_cfg():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = edict()
    cfg.seed = 123
    cfg.world_size = 1
    cfg.use_ddp = False
    cfg.use_apex = False
    cfg.num_workers = 8

    # parameters for simclr
    cfg.simclr = edict()
    cfg.simclr.normalize = True
    cfg.simclr.projection_dim = 64

    cfg.cl = edict()

    cfg.cl.dataset = edict()
    cfg.cl.dataset.name = 'CIFAR10'
    cfg.cl.dataset.root = Path("/home/gauenk/data/cifar10/")
    cfg.cl.dataset.n_classes = 10
    cfg.cl.dataset.transforms = edict()
    cfg.cl.dataset.transforms.low_light = False

    cfg.cl.batch_size = 128
    cfg.cl.image_size = 32
    cfg.cl.workers = 1
    cfg.cl.device = device
    cfg.cl.global_step = 0
    cfg.cl.epochs = 100

    cfg.cl.optim = edict()
    cfg.cl.optim.adam = edict()
    cfg.cl.optim.adam.lr = 3e-4

    cfg.cl.model_path = Path('./output')
    cfg.cl.temperature = 0.5
    cfg.cl.load = False
    cfg.cl.resnet = 'resnet50'
    cfg.cl.start_epoch = 0
    cfg.cl.checkpoint_interval = 1
    cfg.cl.log_interval = 1
    cfg.cl.current_epoch = 0


    cfg.cls = edict()

    cfg.cls.dataset = edict()
    cfg.cls.dataset.name = 'CIFAR10'
    cfg.cls.dataset.root = Path("/home/gauenk/data/cifar10/")
    cfg.cls.dataset.n_classes = 10
    cfg.cls.dataset.download = False

    cfg.cls.batch_size = 128
    cfg.cls.image_size = 32
    cfg.cls.workers = 1
    cfg.cls.device = device
    cfg.cls.global_step = 0
    cfg.cls.epochs = 100

    cfg.cls.optim = edict()
    cfg.cls.optim.adam = edict()
    cfg.cls.optim.adam.lr = 3e-4

    cfg.cls.model_path = './output'
    cfg.cls.load = False
    cfg.cls.checkpoint_interval = 1
    cfg.cls.log_interval = 1
    cfg.cls.current_epoch = 0

    cfg.exp_name = "resnet50_v1"
    cfg.cl.model_path = Path(f"{ROOT_PATH}/output/cl/{cfg.exp_name}/")
    cfg.cls.model_path = Path(f"{ROOT_PATH}/output/cls/{cfg.exp_name}/")

    return cfg
