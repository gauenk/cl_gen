#!/usr/bin/python3.8

# python imports
import os,sys
sys.path.append("./lib")
import argparse,json,uuid,yaml,io
from easydict import EasyDict as edict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

# project imports
import settings
from n2sim.main import run_me

if __name__ == "__main__":
    run_me()
