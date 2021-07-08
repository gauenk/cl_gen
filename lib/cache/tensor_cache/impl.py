"""
Tensor Cache

tensors <-> filenames

read,write,append

"""

# -- python imports --
import numpy as np

# -- pytorch imports --
import torch

def append_tensor_cache(root,fieldname,data,dim=2,overwrite=False):
    path = root / fieldname
    if not path.exists(): path.mkdir(parents=True)
    if overwrite is False:
        names = get_tensor_cache_names(path,data)
        r_data = read_tensor_cache(path,names)
        for name,r_value in data.items():
            data[name] = torch.cat([r_value,data[name]],dim=dim)
    write_tensor_cache(path,data)
    
    return str(path)

def get_tensor_cache_names(path,data):
    names_fn = path / "names.npy"
    if not names_fn.exists():
        if isinstance(data,dict): names = np.array(list(data.keys))
        else: names = None
    else:
        names = np.load(names_fn,allow_pickle=False)        
    return names

