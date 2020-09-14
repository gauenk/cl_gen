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
from datasets import load_dataset

# project code [denoising lib]
from denoising import run_main,load_cfg,save_cfg,get_cfg,get_args

def run_experiment_set():
    args = get_args()
    run_main(args)
    
    
if __name__ == "__main__":
    run_experiment_set()
