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

# -- project imports --
from settings import ROOT_PATH
from lpas.main import get_main_config

# -- [lpas] plotting functions --
from .plots.coupling import plot_frame_index_v_remaining_fixed
from .plots.filters import plot_unet_filters_v_image_content

# -- [explore] imports --
from .mesh import create_mesh,get_setup_fxn
from .io import save_exp,load_exp,clear_cache,get_uuid
from .experiment import execute_experiment
from .utils import append_to_record

def main():

    # -- Load Default Config --
    cfg = get_main_config()
    cfg.explore_package = "lpas" # -- pick package to explore --

    # -- set explore and bss dir --
    cfg.bss_batch_size = 30
    cfg.explore_dir = Path(ROOT_PATH) / f"./output/explore/{cfg.explore_package}/"
    if not cfg.explore_dir.exists(): cfg.explore_dir.mkdir(parents=True)
    cfg.bss_dir = cfg.explore_dir / "./bss/"
    if not cfg.bss_dir.exists(): cfg.bss_dir.mkdir(parents=True)

    # -- data loader config --
    cfg.drop_last = {'tr':True,'val':True,'te':True}

    # -- Pick Experiment Version --
    version = "v3"

    # -- Load Experiment Mesh --
    experiments,order = create_mesh(version)
    config_setup = get_setup_fxn(version)

    # -- Clear Cache --
    if False: clear_cache()

    # -- Run Experiment --
    for config in tqdm(experiments):
        results = load_exp(config)
        uuid = get_uuid(config)
        if results is None:
            exp_cfg = config_setup(cfg,config)
            exp_cfg.uuid = uuid
            results = execute_experiment(exp_cfg)
            save_exp(exp_cfg.uuid,config,results)
    records = load_records(version)

    """
    convert the "filename <-> vec" inside of the uuid interface.

    pros:
    - the "results" are vectors so we can manipulate in obvious ways
    
    cons:
    - more coding. perhaps messy deps among packages
    """
    # -- Create Visualizations of Relevant Data --
    plot_frame_index_v_remaining_fixed(cfg,records,order)
    # plot_unet_filters_v_image_content(cfg,records,order)
    

if __name__ == "__main__":
    main()
    
