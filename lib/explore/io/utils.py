
# -- python imports --
import json,pprint
import uuid as uuid_gen
from pathlib import Path
pp = pprint.PrettyPrinter(depth=5)

# -- [local] project imports --
from .config import cfg
from .tensor_cache import convert_files_to_tensors,convert_tensors_to_files

def write_results_file(path,data):
    print(path,path.parents[0])
    data_files = convert_tensors_to_files(path.parents[0],data)
    with open(path,'w') as f:
        json.dump(data_files,f)

def read_results_file(path):
    with open(path,'r') as f:
        data = json.load(f)
    data_tensors = convert_files_to_tensors(path.parents[0],data)
    return data_tensors

def check_results_exists(uuid):
    path = cfg.root / Path(uuid) / "results.pkl"
    return path.exists()

def print_formatted_exp_info(config,indent=1):
    pp = pprint.PrettyPrinter(depth=5,indent=indent)
    pp.pprint(config)

    
