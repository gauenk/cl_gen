
# -- python imports --
from pathlib import Path

# -- [local] project --
from .config import cfg
from .uuid_cache import get_uuid_from_config
from .utils import read_results_file

def load_exp_list(exp_config_list):
    results_list = []
    for exp_config,results in zip(exp_config_list,results_list):
        results_list.append(load_exp(exp_config,order,results))
    return results_list

def load_exp(config):
    uuid = get_uuid_from_config(config)
    if uuid == -1: return None
    path = cfg.root / Path(uuid)
    if not path.exists(): return None
    path = path / "results.pkl"
    results = read_results_file(path)
    return results
