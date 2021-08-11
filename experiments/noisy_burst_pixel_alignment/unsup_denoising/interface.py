
# -- python imports --
import time,os
from tqdm import tqdm
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from pathlib import Path

# -- multiprocess experiments --
from concurrent.futures import ProcessPoolExecutor as Pool
# from multiprocessing import Pool

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# -- cuda profiler --
import nvtx

# -- project imports --
from pyutils import tile_patches,save_image,torch_to_numpy
from pyutils.vst import anscombe
from patch_search import get_score_function
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

# -- [align] package imports --
import align.nnf as nnf # nnf method
from align.combo.eval_scores import EvalBlockScores # eval method choice
from align.combo.optim import AlignOptimizer # optimizer
from align.xforms import flow_to_pix,pix_to_flow,flow_to_blocks # conversions
from align.xforms import align_from_flow,align_from_pix  # alignments
from align import compute_epe,compute_aligned_psnr # evaluation
from align.xforms import create_isize # misc

# -- [experiment] imports --
import cache_io
from unsup_denoising.experiments import picker
from unsup_denoising._paths import EXP_PATH
import unsup_denoising.experiments.compare_to_competitors as compare_to_competitors

def run():
    # exp_info = picker.run()
    exp_info = compare_to_competitors.get_run_info()
    run_exp(exp_info)

def run_all():
    exp_info_list = picker.get_all_exps()
    for exp_info in exp_info_list:
        run_exp(exp_info)

def run_exp(exp_info):

    # -- Experiment Picker --
    execute_experiment = exp_info['exec']
    plot_experiment = exp_info['plot']
    cache_name = exp_info['cache_name']
    config_name = exp_info['config_name']
    get_cfg_defaults = exp_info['get_cfg_defaults']
    get_exp_cfgs = exp_info['get_exp_cfgs']
    setup_exp_cfg = exp_info['setup_exp_cfg']

    # -- Load Default Config --
    cfg = get_cfg_defaults()
    cfg.gpuid = 1
    cfg.device = f"cuda:{cfg.gpuid}"
    cfg.pid = os.getpid()
    # torch.cuda(device=cfg.gpuid)

    # -- Init Experiment Cache  --
    cache_root = EXP_PATH / cache_name
    cache = cache_io.ExpCache(cache_root,cache_name)
    # cache.clear()

    # -- Load Experiment Mesh --
    experiments,order,exp_grids = get_exp_cfgs(config_name)

    # -- Run experiments --
    exec_exps = {'exec':execute_experiment,'setup':setup_exp_cfg}
    run_experiment_set(cfg,cache,experiments,exec_exps)
    records = cache.load_flat_records(experiments)

    # -- g-75p0 and pn-20p0 -> {std:75,alpha:-1},{std:-1,alpha:20}, respectively --
    expand_noise_nums(records)

    # -- psnrs,epes_of,epes_nnf means --
    fields = ['psnrs','epes_of','epes_nnf']
    compute_field_means(records,fields)

    # -- Run Plots --
    plot_experiment(records,exp_grids)

def run_experiment_set(cfg,cache,experiments,exec_exps):
    PARALLEL = True
    if PARALLEL:
        return run_experiments_set_parallel(cfg,cache,experiments,exec_exps)
    else:
        return run_experiments_set_serial(cfg,cache,experiments,exec_exps)

def run_experiments_set_serial(cfg,cache,experiments,exec_exps):

    nexps = len(experiments)
    for exp_num,config in enumerate(experiments):
        print("-="*25+"-")
        print(f"Running exeriment number {exp_num+1}/{nexps}")
        print("-="*25+"-")
        print(config)
        results = cache.load_exp(config)
        uuid = cache.get_uuid(config)
        if results is None:
            exp_cfg = exec_exps['setup'](cfg,config)
            exp_cfg.uuid = uuid
            results = exec_exps['exec'](exp_cfg)
            cache.save_exp(exp_cfg.uuid,config,results)

# -- wrap experiment --
def wrap_execute_experiment(inputs):
    cfg,config,uuid,exec_exps = inputs
    exp_cfg = exec_exps['setup'](cfg,config)
    exp_cfg.uuid = uuid
    results = exec_exps['exec'](exp_cfg)
    return uuid,config,results

def run_experiments_set_parallel(cfg,cache,experiments,exec_exps):

    # -- 1.) get experiments to run --
    torun_configs = []
    for exp_num,config in enumerate(experiments):    
        results = cache.load_exp(config)
        uuid = cache.get_uuid(config)
        if results is None: torun_configs.append([cfg,config,uuid,exec_exps])
            
    # -- 2.) batch experiments using pool --
    max_jobs = 4
    pool = Pool(max_jobs)
    # p_results = pool.imap_unordered(wrap_execute_experiment,torun_configs)
    p_results = pool.map(wrap_execute_experiment,torun_configs)
    for exp_uuid,exp_config,exp_results in p_results:
        cache.save_exp(exp_uuid,exp_config,exp_results)
    return

def compute_field_means(records,fields):
    for field in fields:
        means,stds = [],[]
        for elem in records[field]:
            means.append(np.mean(elem))
            stds.append(np.std(elem))
        records[f'{field}_mean'] = means
        records[f'{field}_std'] = stds

def expand_noise_nums(records):
    ntype_series = records['noise_type']
    stds = []
    alphas = []
    for ntype_elem in ntype_series:
        ntype = ntype_elem.split('-')[0]
        nlevel = ntype_elem.split('-')[1]
        if ntype == "g":
            std = float(nlevel.replace("p","."))
            stds.append(std)
            alphas.append(-1)
        elif ntype == "pn":
            alpha = float(nlevel.split('-')[1].replace("p","."))
            stds.append(-1)
            alphas.append(alpha)
        else:
            raise ValueError(f"Uknown noise type [{ntype}]")
    records['std'] = stds
    records['alpha'] = alphas
