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
from n2n.stn_n2n import run_me as run_me_stn

if __name__ == "__main__":
    run_me_stn()
