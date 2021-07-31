
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
from noisy_alignment.experiments import picker
from noisy_alignment._paths import EXP_PATH

def run():
    exp_info = picker.run()
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
    cache_root = EXP_PATH
    cache = cache_io.ExpCache(cache_root,cache_name)
    cache.clear()

    # -- Load Experiment Mesh --
    cfg_version = "1"
    experiments,order = get_exp_cfgs(config_name)

    # -- Run Experiment --
    for config in tqdm(experiments):
        results = cache.load_exp(config)
        uuid = cache.get_uuid(config)
        if results is None:
            exp_cfg = setup_exp_cfg(cfg,config)
            exp_cfg.uuid = uuid
            results = execute_experiment(exp_cfg)
            cache.save_exp(exp_cfg.uuid,config,results)
    records = cache.load_records(experiments)
    
    # -- Run Plots --
    plot_experiment(records)
