
# -- python imports --
import numpy as np
from pathlib import Path

# -- pytorch imports --
import torch

# -- local imports --
from ._read import read_tensor_cache
from ._write import write_tensor_cache
from ._utils import get_tensor_cache_names

def append_tensor_cache(path,data,dim=2,overwrite=False):
    if not path.exists(): path.mkdir(parents=True)
    if overwrite is False:
        names = get_tensor_cache_names(path,data)
        r_data = read_tensor_cache(path,names)
        for name,r_value in data.items():
            data[name] = torch.cat([r_value,data[name]],dim=dim)
    write_tensor_cache(path,data)
