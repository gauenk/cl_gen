"""
the main function to 
- create a dataset exploring hyperparameters with many hyperparameters
- versions categorize the inputs and outputs

"""

# -- python imports --
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# -- pytorch imports --
import torch

# -- project imports --
from settings import ROOT_PATH
from lpas.main import get_main_config

# -- [lpas] plotting functions --
from n2a.plots.domain_adaptation import plot_noise_levels,plot_noise_intervals

# -- [explore] imports --
import cache_io
from n2a.picker import run_experiment_picker
from n2a.mesh import create_mesh,get_setup_fxn
from n2a.exps import execute_nnf_experiment,execute_unsup_experiment

def run_me():

    # -- Load Default Config --
    cfg = get_main_config()
    cfg.gpuid = 0
    cfg.device = f"cuda:{cfg.gpuid}"
    # torch.cuda(device=cfg.gpuid)

    # -- data loader config --
    cfg.drop_last = {'tr':True,'val':True,'te':True}

    # -- Init Experiment Cache  --
    cache_root = Path("./output/n2a/exp_cache/")
    cache_version = "1"
    cache = cache_io.ExpCache(cache_root,cache_version)
    cache.clear()

    # -- Load Experiment Mesh --
    mesh_version = "1"
    experiments,order = create_mesh(mesh_version)
    config_setup = get_setup_fxn(mesh_version)

    # -- Run Experiment --
    for config in tqdm(experiments):
        results = cache.load_exp(config)
        uuid = cache.get_uuid(config)
        if results is None:
            exp_cfg = config_setup(cfg,config)
            exp_cfg.uuid = uuid
            results = execute_nnf_experiment(exp_cfg)
            cache.save_exp(exp_cfg.uuid,config,results)
    records = cache.load_records(experiments)
    

if __name__ == "__main__":
    main()
