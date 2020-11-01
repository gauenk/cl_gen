"""
This code implements the code to run denoising experiments.
This code allows for multiple-gpu training.




This code cleans up example_static.py

todo:

- run the "testing" script on the third gpu asynchronously

"""

# python code
import os,sys
sys.path.append("./lib")
import tempfile
from easydict import EasyDict as edict
from pathlib import Path

# torch code
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# project code
import settings
from datasets import load_dataset

# project code [simcl lib]
from simcl.train import run_train
from simcl.test import run_test
from simcl.config import get_cfg,get_args
from .model_io import load_model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9998'
    # initialize the process group
    proc_group = dist.init_process_group("nccl",
                                   # init_method='file://srv/nfs/sharedfile',
                                   init_method='tcp://192.168.2.10:9998',
                                   rank=rank,
                                   world_size=world_size)
    return proc_group

def cleanup():
    dist.destroy_process_group()
    
#
# Experiment setup and execution
#
    
def run_experiment(rank, cfg):
    print("Simple Contrastive Learning")
    print(f"Running DDP experiment on rank {rank}")

    # bandaid for now.
    cfg.denoising_prep = False
    dsname = "cifar10"
    cfg.summary_log_dir = Path(f"{settings.ROOT_PATH}/runs/simcl/{dsname}/{cfg.exp_name}/")
    cfg.test_interval = 1000
    cfg.val_interval = 1000

    proc_group = None
    if cfg.use_ddp:
        proc_group = setup(rank,cfg.world_size)
    cfg.device = 'cuda:%d' % rank
    cfg.rank = rank
    torch.cuda.set_device(rank)
    
    # load the data
    cfg.rank = rank # only for data loading!
    data,loader = load_dataset(cfg,'simcl')

    # load models    
    model = load_model(cfg,rank,proc_group)

    # run experiment
    if cfg.mode == "train": 
        run_train(cfg,rank,model,data,loader)
    elif cfg.mode == "test":
        run_test(cfg,rank,model,data,loader)
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
    cfg.activation_hooks = False

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
    cfg.activation_hooks = False

    # update name with "_localized" info
    # cfg.exp_name += "_localized"
    # dsname = cfg.dataset.name.lower()
    # cfg.model_path = Path(f"{settings.ROOT_PATH}/output/simcl/{dsname}/{cfg.exp_name}/model/")
    # cfg.optim_path = Path(f"{settings.ROOT_PATH}/output/simcl/{dsname}/{cfg.exp_name}/optim/")
    # cfg.summary_log_dir = Path(f"{settings.ROOT_PATH}/runs/simcl/{dsname}/{cfg.exp_name}/")

    if gpuid is None: gpuid = 0
    run_experiment(gpuid,cfg)

if __name__ == "__main__":
    main()
