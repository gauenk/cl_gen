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


# project code
from pyutils.timer import Timer
from datasets import load_dataset

# project code [simcl lib]
from simcl import run_ddp,load_cfg,save_cfg,get_cfg,get_args
from simcl.exp_set_v1 import get_experiment_set_v1
from simcl.exp_set_v2 import get_experiment_set_v2
from simcl.exp_utils import record_experiment,_build_v1_summary
# from simcl.test_mnist import run_test as run_mnist_test

def run_experiment_set(version="v1"):

    # get experiment setups
    if version == "v1":
        cfgs = get_experiment_set_v1()
    elif version == "v2":
        cfgs = get_experiment_set_v2()
    else:
        raise ValueError(f"Unknown version [{version}]")

    # train grid
    run_experiment_serial(cfgs,version,"train")

    # test grid
    # run_experiment_mode(cfgs,"test")

#
# Run a single experiment on many (or one) gpus
#

def run_experiment_serial(cfgs,version,mode,use_ddp=True):
    for cfg in cfgs:
        t = Timer()

        # log process
        record_experiment(cfg,version,mode,'start',t)
        print(_build_v1_summary(cfg))

        cfg.mode = mode
        if cfg.mode == "test":
            cfg.load = True
            cfg.epoch_num = cfg.epochs # load last epoch
        else:
            cfg.load = False
            cfg.epoch_num = -1
            # cfg.load = True
            # cfg.epoch_num = 500

        if use_ddp:
            # cfg.use_apex = False
            run_ddp(cfg=cfg)
        else:
            run_serial(cfg=cfg)            

        record_experiment(cfg,version,mode,'start',t)

def run_default():
    args = get_args()
    run_dpp(args=args)
    
if __name__ == "__main__":
    # run_mnist_test()
    # run_default()
    # run_experiment_set("v1")
    run_experiment_set("v2")
