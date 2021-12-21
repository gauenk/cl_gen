
# -- python imports --
import numpy as np
from pathlib import Path

# -- pytorch imports --
import torch

def write_tensor_cache(path,data):
    if isinstance(data,dict):
        write_dict_tensor_cache(path,data)
    else:
        tensor_path = path / "tensor.npy"
        if torch.is_tensor(data): data = data.numpy()
        np.save(tensor_path,data,allow_pickle=True)

def write_dict_tensor_cache(path,data):
    names_fn = path / "names.npy"
    np.save(names_fn,np.array(list(data.keys())),allow_pickle=True)
    # print(data)
    for name,value in data.items():
        if isinstance(value,list):
            if torch.is_tensor(value[0]):
                value = torch.stack(value,dim=0)
            else:
                value = torch.Tensor(value)
            if isinstance(value[0],np.ndarray):
                value = np.stack(value,axis=0)
            else:
                value = np.array(value)
        name_path = path / f"{name}.npy"
        if torch.is_tensor(value): value = value.numpy()
        np.save(name_path,value,allow_pickle=True)

__all__ = ['write_tensor_cache','write_dict_tensor_cache']
