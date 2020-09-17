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

# project code [denoising lib]
from denoising import run_ddp,load_cfg,save_cfg,get_cfg,get_args
from denoising.exp_set_v1 import get_experiment_set_v1
from denoising.exp_utils import record_experiment,_build_v1_summary
from denoising.test_mnist import run_test as run_mnist_test

def run_experiment_set_v1():

    # get experiment setups
    cfgs = get_experiment_set_v1()

    # train grid
    run_experiment_serial(cfgs,"train")

    # test grid
    # run_experiment_mode(cfgs,"test")


#
# Run a single experiment on many (or one) gpus
#

def run_experiment_serial(cfgs,mode,use_ddp=True):
    for cfg in cfgs:
        t = Timer()
        record_experiment(cfg,f'v1_{mode}','start',t)
        print(_build_v1_summary(cfg))
        cfg.mode = mode
        if cfg.mode == "test":
            cfg.load = True
            cfg.epoch_num = cfg.epochs # load last epoch
        else:
            cfg.load = False
            cfg.epoch_num = -1
        if use_ddp:
            run_ddp(cfg=cfg)
        else:
            run_serial(cfg=cfg)            
        record_experiment(cfg,f'v1_{mode}','end',t)

def run_default():
    args = get_args()
    run_dpp(args=args)
    
if __name__ == "__main__":
    # run_mnist_test()
    # run_default()
    run_experiment_set_v1()
