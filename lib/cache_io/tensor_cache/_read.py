
# -- python imports --
import numpy as np
from pathlib import Path

# -- pytorch imports --
import torch

def read_tensor_cache(path,names):
    if names is None:
        tensor_path = path / "tensor.npy"
        if tensor_path.exists(): return np.load(tensor_path,allow_pickle=True)
        else: return torch.Tensor([])
    else:
        return read_dict_tensor_cache(path,names)

def read_dict_tensor_cache(path,names):
    data = {}
    for name in names:
        names_fn = path / f"{name}.npy"
        if not names_fn.exists(): value = []
        else: value = np.load(names_fn,allow_pickle=True)
        try:
            data[name] = torch.tensor(value)
        except:
            data[name] = value.tolist()
    return data

__all__ = ['read_tensor_cache','read_dict_tensor_cache']

