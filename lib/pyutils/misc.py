
# python imports
import numpy as np
import pickle,sys,os,yaml,io
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
from itertools import chain, combinations

# this is the only allowed project import in this file.
import settings

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


def create_combination(indices,start,end):
    cfi = chain.from_iterable
    subset = cfi(combinations(list(indices) , r+1 ) for r in range(start,end))
    subset = np.array([np.array(elem) for elem in list(subset)])
    return subset

