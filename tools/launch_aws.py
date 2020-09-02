"""
Launch a few models on AWS for faster training.

"""

import argparse
import subprocess
import sys,os
sys.path.append("./lib/")

# project imports
import settings
from datasets import get_dataset
from pyutils.cfg import get_cfg

def get_args():
    parser = argparse.ArgumentParser(description="Launch a series of processes on AWS")
    parser.add_argument("--dataset", type=str, default="MNIST",
                        help="dataset to run experiments")
    args = parser.parse_args()
    return args
    
def wait(procs):
    # wait for all to finish
    for p in procs:
        p.wait()

def run_process(dataset,noise_level,N,bs,i):
    pycmd = ["python3","./tools/aws_denoising.py"]
    pycmd += ["--noise-level","{:2.3e}".format(noise_level)]
    pycmd += ["--N","{:d}".format(N)]
    pycmd += ["--dataset",dataset]
    pycmd += ["--batch-size","{:d}".format(bs)]
    pycmd += ["--gpuid","{:d}".format(gpui)]
    print("Running: {}".format(' '.join(pycmd)))
    return subprocess.Popen(pycmd)

def run_grid(dataset):

    download_dataset(dataset)
    Ngrid = [2, 5, 10, 15]
    noise_levels = [5e-2, 1e-1, 5e-1]
    bsizes = [400, 200, 100, 75]

    ngpus = 4
    gpuid = 0
    for i,(N,bs) in enumerate(zip(Ngrid,bsizes)):

        for noise_level in noise_levels:

            # launch experiment
            if len(procs) == 4:
                wait(procs)
                procs = []
            p = run_process(dataset,noise_level,N,bs,gpuid)
            gpuid = (gpuid + 1) % ngpus
            procs.append(p)

        # download results
        for N in Ngrid:
            pass

def download_dataset(dataset):
    cfg = get_cfg()
    cfg.cls.dataset.name = dataset
    cfg.cls.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.cls.dataset.download = True
    get_dataset(cfg,'cls')

if __name__ == "__main__":
    args = get_args()
    run_grid(args.dataset)
