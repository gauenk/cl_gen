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

def get_load_exps(Ngrid,noise_levels):
    exps = {}    
    for N in Ngrid:
        exps[N] = {}
        for noise in noise_levels:
            exps[N][noise] = {}
            exps[N][noise]['epoch'] = 0
            exps[N][noise]['name'] = None
    exps[2][5e-2]['name'] = "36c93317-ddde-4f97-827a-ff14b7fd568f"
    exps[2][5e-2]['epoch'] = 440
    exps[2][5e-1]['name'] = "d4bae353-f95a-40b4-bfec-0cd78f8b58a3"
    exps[2][5e-1]['epoch'] = 438
    exps[2][1e-1]['name'] = "89174fb6-dabc-4d62-93b5-5012e1d94d27"
    exps[2][1e-1]['epoch'] = 435
    exps[5][5e-2]['name'] = "11604220-24f4-445a-9974-dcafdb1f79f8"
    exps[5][5e-2]['epoch'] = 168
    return exps

def run_process(dataset,noise_level,N,bs,i,exp):
    pycmd = ["python3","./tools/aws_denoising.py"]
    pycmd += ["--noise-level","{:2.3e}".format(noise_level)]
    pycmd += ["--N","{:d}".format(N)]
    pycmd += ["--dataset",dataset]
    pycmd += ["--batch-size","{:d}".format(bs)]
    pycmd += ["--gpuid","{:d}".format(i)]
    pycmd += ["--epoch-num","{:d}".format(exp['epoch'])]
    if exp['name'] is not None:
        pycmd += ["--name",exp['name']]
    print("Running: {}".format(' '.join(pycmd)))
    return subprocess.Popen(pycmd)

def run_grid(dataset):

    download_dataset(dataset)
    Ngrid = [2, 5, 10, 15]
    noise_levels = [5e-2, 1e-1, 5e-1]
    bsizes = [400, 200, 100, 75]
    exps = get_load_exps(Ngrid,noise_levels)

    ngpus = 4
    gpuid = 0
    procs = []
    for i,(N,bs) in enumerate(zip(Ngrid,bsizes)):

        for noise_level in noise_levels:

            # launch experiment
            if len(procs) == 4:
                wait(procs)
                procs = []
            exp = exps[N][noise_level]
            p = run_process(dataset,noise_level,N,bs,gpuid,exp)
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
