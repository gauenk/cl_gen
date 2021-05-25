"""
Tensor Cache

tensors <-> filenames

read,write,append

"""

# -- python imports --
import numpy as np

# -- pytorch imports --
import torch

def convert_files_to_tensors(root,results):
    tensors = {}
    for field,field_data in results.items():
        path =  root / field
        names = get_tensor_cache_names(path,field_data)
        data = read_tensor_cache(path,names)
        tensors[field] = data
    return tensors

def convert_tensors_to_files(root,results):
    files = {}
    for field,field_data in results.items():
        path = append_tensor_cache(root,field,field_data,overwrite=True)
        files[field] = path
    return files

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

def get_tensor_cache_names(path,data):
    names_fn = path / "names.npy"
    if not names_fn.exists():
        if isinstance(data,dict): names = np.array(list(data.keys))
        else: names = None
    else:
        names = np.load(names_fn,allow_pickle=False)        
    return names

def read_tensor_cache(path,names):
    if names is None:
        tensor_path = path / "tensor.npy"
        if tensor_path.exists(): return np.load(tensor_path,allow_pickle=False)
        else: return torch.Tensor([])
    else:
        return read_dict_tensor_cache(path,names)

def read_dict_tensor_cache(path,names):
    data = {}
    for name in names:
        names_fn = path / f"{name}.npy"
        if not names_fn.exists(): value = []
        else: value = np.load(names_fn,allow_pickle=False)
        data[name] = torch.tensor(value)
    return data

    
