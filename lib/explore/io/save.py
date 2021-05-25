
# -- python imports --
from pathlib import Path

# -- [local] project --
from .config import cfg
from .uuid_cache import get_uuid_from_config
from .utils import write_results_file,check_results_exists


verbose = True

def save_exp_list(config_list,results_list,overwrite=False):
    for config,results in zip(config_list,results_list):
        save_exp(config,results,overwrite=overwrite)

def save_exp(uuid,config,results,overwrite=False):
    check_uuid = get_uuid_from_config(config)
    assert check_uuid == -1 or uuid == check_uuid, "Only one uuid per config." 
    exists = check_results_exists(uuid)
    if overwrite is True or exists is False:
        if (exists is True) and verbose:
            print("Overwriting Old UUID.")
        if verbose: print(f"UUID [{uuid}]")
        path = cfg.root / Path(uuid)
        if not path.exists(): path.mkdir(parents=True)
        path = path / "results.pkl"
        write_results_file(path,results)
    else:
        print(f"WARNING: Not writing. UUID [{uuid}] exists.")


