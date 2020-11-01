# python imports
import os,sys
sys.path.append("./lib")
import argparse,json,uuid,yaml,io
from easydict import EasyDict as edict
from pathlib import Path

# project imports
import settings
from denoise_gan.main import run_me
from denoise_gan.main_simclr import run_me as run_me_simclr
from denoise_gan.main_blind import run_me as run_me_blind

if __name__ == "__main__":
    # run_me()
    # run_me_simclr()
    run_me_blind()
