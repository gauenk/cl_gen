
# -- python imports --
import numpy as np
from pathlib import Path

# -- pytorch imports --
import torch


__all__ = [write_tensor_cache,write_dict_tensor_cache]

def write_tensor_cache(path,data):
    if isinstance(data,dict):
        write_dict_tensor_cache(path,data)
    else:
        tensor_path = path / "tensor.npy"
        np.save(tensor_path,data.numpy(),allow_pickle=False)

def write_dict_tensor_cache(path,data):
    names_fn = path / "names.npy"
    np.save(names_fn,np.array(list(data.keys())),allow_pickle=False)
    for name,value in data.items():
        name_path = path / f"{name}.npy"
        np.save(name_path,value.numpy(),allow_pickle=False)

