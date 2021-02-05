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
from n2n.main import run_me,run_me_Ngrid
from n2n.n2n_main import run_me as run_me_n2n
from n2n.kpn_n2n import run_me as run_me_kpn
from n2n.kpn_n2n import run_me_grid as run_me_grid_kpn
from n2n.dncnn import run_me as run_me_dncnn

if __name__ == "__main__":
    run_me()
    # run_me_Ngrid()
    # run_me_n2n()
    # run_me_kpn()
    # run_me_grid_kpn()
    run_me_dncnn()
