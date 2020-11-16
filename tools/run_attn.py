# python imports
import os,sys
sys.path.append("./lib")
import argparse,json,uuid,yaml,io
from easydict import EasyDict as edict
from pathlib import Path

# project imports
import settings
from attn.main import run_me

if __name__ == "__main__":
    run_me()
