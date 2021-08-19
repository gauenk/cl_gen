
# -- python imports --
import time
from tqdm import tqdm
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from pathlib import Path

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

# -- [experiment] imports --
import cache_io
from noisy_alignment.experiments import picker
from noisy_alignment._paths import EXP_PATH

import noisy_alignment.experiments.check_boot_boost as check_boot_boost
import noisy_alignment.experiments.compare_to_theory as compare_to_theory
import noisy_alignment.experiments.compare_to_competitors as compare_to_competitors
import noisy_alignment.experiments.stress_tests as stress_tests


def run():
    
    # exp_info = picker.run()
    # exp_info = compare_to_competitors.get_run_info()
    exp_info = check_boot_boost.get_run_info()
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
    cfg = get_cfg_defaults() # todo
    cfg.gpuid = 0
    cfg.device = f"cuda:{cfg.gpuid}"
    # torch.cuda(device=cfg.gpuid)

    # -- Init Experiment Cache  --
    # cache_name += "_v2"
    cache_root = EXP_PATH / cache_name
    cache = cache_io.ExpCache(cache_root,cache_name)
    # cache.clear()

    # -- Load Experiment Mesh --
    experiments,order,exp_grids = get_exp_cfgs(config_name)

    # -- Run Experiment --
    exp_cfgs = []
    for config in tqdm(experiments,total=len(experiments)):
        results = cache.load_exp(config)
        uuid = cache.get_uuid(config)
        print(uuid)
        exp_cfg = setup_exp_cfg(cfg,config)
        exp_cfg.uuid = uuid
        exp_cfgs.append(exp_cfg)
        if results is None:
            results = execute_experiment(exp_cfg)
            for key,val in results.items():
                print("r",key,val.shape)
            cache.save_exp(exp_cfg.uuid,config,results)
    records = cache.load_flat_records(experiments)
    # print(records)
    print(records.columns)

    # -- g-75p0 and pn-20p0 -> {std:75,alpha:-1},{std:-1,alpha:20}, respectively --
    expand_noise_nums(records)

    # -- psnrs,epes_of,epes_nnf means --
    fields = ['psnrs','epes_of','epes_nnf']
    compute_field_means(records,fields)
    
    # frecords = records[records['methods'].isin(['ave','est','split'])]
    frecords = records[records['methods'].isin(['est','ave'])]
    for img_index,igroup in frecords.groupby('image_index'):
        print(img_index)
        for nframes,fgroup in igroup.groupby('nframes'):
            print(nframes)
            print(fgroup[['std','patchsize','psnrs_mean','methods','random_seed']])

    # for elem in records[['std','bsname','psnrs_mean']].iterrows():
    #     print(elem)


    # -- Run Plots --
    plot_experiment(records,exp_grids,exp_cfgs)

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
