
# python imports
import numpy as np
import numpy.random as npr
import pickle,sys,os,yaml,io
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
from itertools import chain, combinations
import operator as op
from functools import reduce

# this is the only allowed project import in this file.
import settings

def torch_to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    else:
        return tensor

def dict_torch_to_numpy(dict_tensors):
    dict_ndarrays = edict()
    for name,tensor in dict_tensors.items():
        dict_ndarrays[name] = torch_to_numpy(tensor)
    return dict_ndarrays

def edict_torch_to_numpy(dict_tensors):
    edict_ndarrays = edict()
    for name,tensor in dict_tensors.items():
        edict_ndarrays[name] = torch_to_numpy(tensor)
    return edict_ndarrays

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def np_divide(np_array_a,np_array_b):
    not_np = False
    if type(np_array_a) is not np.ndarray:
        not_np = True
        if type(np_array_a) is not list:
            np_array_a = [np_array_a]
        np_array_a = np.array(np_array_a)
    ma_a = np.ma.masked_array(np_array_a)
    div = (ma_a / np_array_b).filled(0)
    if not_np:
        div = div[0]
    return div

def np_log(np_array):
    if type(np_array) is not np.ndarray:
        if type(np_array) is not list:
            np_array = [np_array]
        np_array = np.array(np_array)
    return np.ma.log(np_array).filled(-np.infty)

def write_pickle(data,fn,verbose=False):
    if verbose or settings.verbose >= 2: print("Writing pickle to [{}]".format(fn))
    with open(fn,'wb') as f:
        pickle.dump(data,f)
    if verbose: print("Save successful.")

def read_pickle(fn,verbose=False):
    if verbose or settings.verbose >= 2: print("Reading pickle file [{}]".format(fn))
    data = None
    with open(fn,'rb') as f:
        data = pickle.load(f)
    if verbose: print("Load successful.")
    return data

def get_model_epoch_info(cfg):
    if cfg.load:
        return 0,cfg.epoch_num+1
    else: return 0,0

def write_cfg(cfg,fpath):
    with io.open(fpath,'w',encoding='utf8') as f:
        yaml.dump(cfg,f,default_flow_style=False,allow_unicode=True)
        
def read_cfg(fpath):
    with open(fpath,'r') as f:
        cfg = yaml.load(f,Loader=yaml.Loader)
    return cfg

#
# Combinations
#

def create_combination(indices,start,end):
    cfi = chain.from_iterable
    subset = cfi(combinations(list(indices) , r+1 ) for r in range(start,end))
    subset = np.array([np.array(elem) for elem in list(subset)])
    return subset

def create_subset_grids_fixed(subN,indices,max_subset_size):
    return create_subset_grids(subN-1,subN,indices,max_subset_size)

def create_subset_grids(nmin,nmax,indices,max_subset_size):
    subsets_idx = create_combination(indices,nmin,nmax)
    if subsets_idx.shape[0] > max_subset_size: 
        indices = torch.randperm(subsets_idx.shape[0])[:max_subset_size]
        subsets_idx = subsets_idx[indices]
    return subsets_idx

def sample_subset_grids(num_samples,N):
    samples = npr.choice(N,size=(num_samples,N),replace=True)
    return samples

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2
