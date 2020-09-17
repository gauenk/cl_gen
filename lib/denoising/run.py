"""
This code implements the code to run denoising experiments.
This code allows for multiple-gpu training.




This code cleans up example_static.py

"""

# python code
import os,sys
sys.path.append("./lib")
import tempfile
from easydict import EasyDict as edict

# torch code
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# project code
from datasets import load_dataset


# project code [denoising lib]
from denoising.train import run_train
from denoising.test import run_test
from denoising.config import load_cfg,save_cfg,get_cfg,get_args

from .model_io import load_models
from .optim_io import load_optimizer
from .scheduler_io import load_scheduler
from .config import load_cfg,save_cfg,get_cfg,get_args
from .utils import load_hyperparameters,extract_loss_inputs
from .train import run_train
from .test import run_test


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9999'
    # initialize the process group
    proc_group = dist.init_process_group("nccl",
                                   # init_method='file://srv/nfs/sharedfile',
                                   init_method='tcp://192.168.2.10:9999',
                                   rank=rank,
                                   world_size=world_size)
    return proc_group

def cleanup():
    dist.destroy_process_group()
    
#
# Experiment setup and execution
#
    
def run_experiment(rank, cfg):
    print(f"Running DDP experiment on rank {rank}")
    proc_group = None
    if cfg.use_ddp:
        proc_group = setup(rank,cfg.world_size)
    cfg.device = 'cuda:%d' % rank
    cfg.rank = rank
    torch.cuda.set_device(rank)
    
    # load the data
    cfg.rank = rank # only for data loading!
    data,loader = load_dataset(cfg,'denoising')

    # load models    
    models = load_models(cfg,rank,proc_group)

    # run experiment
    if cfg.mode == "train": 
        run_train(cfg,rank,models,data,loader)
    elif cfg.mode == "test":
        run_test(cfg,rank,models,data,loader)
    else:
        raise ValueError(f"Uknown mode [{cfg.mode}]")
    
def run_ddp(cfg=None,args=None):
    if not args is None and not cfg is None:
        raise InputError("cfg and args cann't both be not None. Pick one!")
    if args is None and cfg is None:
        args = get_args()
    elif cfg is None:
        cfg = get_cfg(args)
    cfg.use_ddp = True
    mp.spawn(run_experiment, nprocs=cfg.world_size, args=(cfg,))
    
def run_localized(cfg=None,args=None,gpuid=None):
    if not args is None and not cfg is None:
        raise InputError("cfg and args cann't both be not None. Pick one!")
    if args is None and cfg is None:
        args = get_args()
    elif cfg is None:
        cfg = get_cfg(args)
    cfg.use_ddp = False
    cfg.world_size = 1
    if gpuid is None: gpuid = 0
    run_experiment(gpuid,cfg)

if __name__ == "__main__":
    main()
