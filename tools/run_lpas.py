#!/usr/bin/python3.8

# python imports
import os,sys
sys.path.append("./lib")
import argparse,json,uuid,yaml,io
from easydict import EasyDict as edict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

# pytorch imports
import torch.multiprocessing as mp


# project imports
import settings
from lpas.main import run_me

if __name__ == "__main__":
    mp.set_start_method('spawn')
    run_me()
