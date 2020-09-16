
"""
todo
check_exp_cache('v1')
load_exp_cache('v1')
save_exp_cache('v1')
record_experiment
"""

# python imports
from pathlib import Path

# project imports
import settings
from pyutils.misc import write_cfg,read_cfg

def get_exp_cache_fn(v):
    logdir = Path(f"{settings.ROOT_PATH}/cache")
    subdir = Path("run_denoising")
    fn = Path(f"{v}.yml")
    path = logdir / subdir
    if not path.exists():
        path.mkdir(parents=True)
    fullpath = path / fn
    return fullpath

def check_exp_cache(v):
    fpath = get_exp_cache_fn(v)
    return fpath.exists()

def load_exp_cache(v):
    fpath = get_exp_cache_fn(v)
    return read_cfg(fpath)

def save_exp_cache(exps,v):
    fpath = get_exp_cache_fn(v)
    write_cfg(exps,fpath)

def get_record_experiment_path(cfg,v):
    logdir = Path(f"{settings.ROOT_PATH}/logs")
    subdir = Path(f"run_denoising/{v}/")
    fulldir = logdir / subdir    
    if not fulldir.exists():
        fulldir.mkdir(parents=True)
    return fulldir

def record_experiment(cfg,v,when,clock):
    rec_path = get_record_experiment_path(cfg,v)
    path = rec_path / Path(f"{cfg.exp_name}.txt")
    if when == "start":
        ftype = "w+" # clear and write
        _log_summary_exp_v1(cfg,path)
        clock.tic()
        writing = "\n\n" + "-="*20 + "\n"
        writing += f"starting {cfg.exp_name}\n"
    elif when == "end":
        ftype = "a+" # append
        clock.toc()        
        writing = f"ending {cfg.exp_name}\n"
        writing += str(clock)
        writing += "\n"
        writing += "-="*20 + "\n"
    else:
        raise ValueError(f'Unknown "when" time [{when}]')
    with open(path,"a+") as f:
        f.write(writing)

#
# Recording for experiment v1
#
        
def _log_summary_exp_v1(exp,fpath):
    summ = _build_v1_summary(exp)
    with open(fpath,'a+') as f:
        f.write(summ)

def _build_v1_summary(cfg):
    summ = f"exp_name: {cfg.exp_name}\n"
    summ += f"noise_params['g']: {cfg.noise_params}\n"
    summ += f"N: {cfg.N}\n"
    summ += f"agg_enc_fxn: {cfg.agg_enc_fxn}\n"
    summ += f"hyper_params: {cfg.hyper_params}\n"
    return summ


