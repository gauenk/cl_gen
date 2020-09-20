# python imports
import os,sys,time
sys.path.append("./lib")
import subprocess

# project imports
import settings
from pyutils.timer import Timer
from denoising.exp_set_v1 import get_experiment_set_v1
from denoising.exp_set_v2 import get_experiment_set_v2
from denoising.exp_utils import record_experiment,_build_summary

#
# Run many experiments on their own gpu
#

def run_experiment_set(version='v1'):

    # get experiment setups
    if version == "v1":
        cfgs = get_experiment_set_v1(True)
    elif version == "v2":
        cfgs = get_experiment_set_v2()
    else:
        raise ValueError(f"Unknown verions [{version}]")
        
    # train grid
    run_experiment_parallel(cfgs,version,"train")

    # test grid
    # run_experiment_parallel(cfgs,version,"test")

def wait(procs):
    for p in procs:
        rcode = p.wait()
        print(rcode,p)

def run_process(cache,cache_id,gpuid):
    pycmd = ["python3.8","./tools/run_single_denoising.py"]
    pycmd += ["--cache",str(cache)]
    pycmd += ["--id",str(cache_id)]
    pycmd += ["--gpuid",str(gpuid)]
    # pycmd += ["--init-lr",'5e-4']
    print("Running: {}".format(' '.join(pycmd)))
    return subprocess.Popen(pycmd)

def run_experiment_parallel(cfgs,version,mode,use_ddp=False):

    ngpus = 2
    nproc_per_gpu = 2
    max_procs = nproc_per_gpu * ngpus
    gpuid = 0
    procs = []

    for idx,cfg in enumerate(cfgs):
        t = Timer()

        # wait if need to
        if len(procs) == max_procs:
            wait(procs)
            procs = []

        # run in proper mode
        cfg.use_ddp = False
        if cfg.mode == "test":
            cfg.load = True
            cfg.epoch_num = cfg.epochs # load last epoch
        else:
            cfg.load = False
            cfg.epoch_num = -1

        # log process
        record_experiment(cfg,f'{version}_{mode}','start',t)

        # launch process
        p = run_process(version,idx,gpuid)

        # print what process is being launched
        print(_build_summary(cfg,version))

        # create separate tensorboard files
        time.sleep(5)

        # update current gpuid
        gpuid = (gpuid + 1) % ngpus

        # add process to list
        procs.append(p)

    # wait if need to
    if len(procs) > 0:
        wait(procs)
        procs = []

if __name__ == "__main__":
    run_experiment_set(version='v1')
    # run_experiment_set(version='v2')

