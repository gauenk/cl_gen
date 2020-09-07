"""
Launch a few models on AWS for faster training.

"""

import argparse
import subprocess
import sys,os,time
sys.path.append("./lib/")

# project imports
import settings
from datasets import get_dataset
from pyutils.cfg import get_cfg

def get_args():
    parser = argparse.ArgumentParser(description="Launch a series of processes on AWS")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="dataset to run experiments")
    parser.add_argument("--mode", type=str, default="train",
                        help="do we want to train or test models?")
    args = parser.parse_args()
    return args
    
def wait(procs):
    # wait for all to finish
    for p in procs:
        rcode = p.wait()
        print(rcode,p)

def get_load_exps(Ngrid,noise_levels):
    exps = {}    
    for N in Ngrid:
        exps[N] = {}
        for noise in noise_levels:
            exps[N][noise] = {}
            exps[N][noise]['epoch'] = 0
            exps[N][noise]['name'] = None
    return exps

def run_process(dataset,mode,noise_level,N,bs,i,exp):
    pycmd = ["python3.8","./tools/aws_denoising.py"]
    pycmd += ["--mode",mode]
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


def run_dataset_grid(dataset,mode,procs,ngpus):

    download_dataset(dataset)

    # Ngrid = [2, 5, 10, 15]
    # noise_levels = [5e-2, 1e-1, 5e-1]
    # bsizes = [400, 200, 100, 75]
    # Ngrid = [2, 7, 15]
    # bsizes = [256, 128, 56]
    Ngrid = [2, 15]
    noise_levels = [5e-1]
    bsizes = [256, 56]
    exps = get_load_exps(Ngrid,noise_levels)

    gpuid = 0
    for i,(N,bs) in enumerate(zip(Ngrid,bsizes)):

        for noise_level in noise_levels:

            # launch experiment
            if len(procs) == ngpus:
                wait(procs)
                procs = []
            exp = exps[N][noise_level]
            p = run_process(dataset,mode,noise_level,N,bs,gpuid,exp)
            time.sleep(5) # create separate tensorboard files
            gpuid = (gpuid + 1) % ngpus
            procs.append(p)

        # download results
        for N in Ngrid:
            pass
    return procs

def run_grid(mode):
    ngpus = 1
    procs = []
    dsnames = ['MNIST','CIFAR10']
    for ds in dsnames:
        procs = run_dataset_grid(ds,mode,procs,ngpus)
    wait(procs)

def download_dataset(dataset):
    cfg = get_cfg()
    cfg.cls.dataset.name = dataset
    dsname = dataset.lower()
    cfg.cls.dataset.root = f"{settings.ROOT_PATH}/data/{dsname}/"
    cfg.cls.dataset.download = True
    get_dataset(cfg,'cls')

if __name__ == "__main__":
    args = get_args()
    # run_dataset_grid(args.dataset,args.mode)
    run_grid(args.mode)

