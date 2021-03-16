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
from nac.main import run_me,run_me_Ngrid

if __name__ == "__main__":
    run_me()

