

from unsup_denoising.experiments.compare_to_competitors.configs import get_cfg_defaults,get_exp_cfgs,setup_exp_cfg
from unsup_denoising.experiments.compare_to_competitors.exp import execute_experiment
from unsup_denoising.experiments.compare_to_competitors.plots import plot_experiment

def get_run_info():
    info = dict()

    info['exec'] = execute_experiment
    info['plot'] = plot_experiment
    info['cache_name'] = "compare_to_competitors"
    info['config_name'] = "compare_to_competitors"
    info['get_cfg_defaults'] = get_cfg_defaults
    info['get_exp_cfgs'] = get_exp_cfgs
    info['setup_exp_cfg'] = setup_exp_cfg

    return info
