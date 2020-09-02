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

def run_process(dataset,noise_level,N,i):
    subprocess.run(["export","CUDA_VISIBLE_DEVICE",f"{i}"])
    pycmd = ["python3.8","./tools/aws_denoising.py"]
    pycmd += ["--noise-level","{:2.3e}".format(noise_level)]
    pycmd += ["--N","{:d}".format(N)]
    pycmd += ["--dataset",dataset]
    print("Running: {}".format(' '.join(pycmd)))
    return subprocess.Popen(pycmd)

def run_grid(dataset):

    download_dataset(dataset)
    noise_levels = [1e-2, 5e-2, 1e-1]
    Ngrid = [2, 5, 10, 20] # 4 gpus

    for noise_level in noise_levels:
        
        procs = []
        for i,N in enumerate(Ngrid):

            # launch experiment
            p = run_process(dataset,noise_level,N,i)
            procs.append(p)

        wait(procs)

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
