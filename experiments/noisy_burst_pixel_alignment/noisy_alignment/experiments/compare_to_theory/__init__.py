

from noisy_alignment.experiments.compare_to_theory.configs import get_cfg_defaults,get_exp_cfgs,setup_exp_cfg
from noisy_alignment.experiments.compare_to_theory.exp import execute_experiment
from noisy_alignment.experiments.compare_to_theory.plots import plot_experiment

def get_run_info():
    info = dict()

    info['exec'] = execute_experiment
    info['plot'] = plot_experiment
    info['cache_name'] = "compare_to_theory"
    info['config_name'] = "compare_to_theory"
    info['get_cfg_defaults'] = get_cfg_defaults
    info['get_exp_cfgs'] = get_exp_cfgs
    info['setup_exp_cfg'] = setup_exp_cfg

    return info

    
