"""

Manage Block Grid Search Space

"""

# -- python imports --
import random
import numpy as np
import numpy.random as npr
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
from datasets.common import get_loader

# -- [local] project imports --
from .utils import get_ref_block_index

verbose = False

def parse_bss(bss_str):
    if bss_str[0:2] == "0m":
        elems = bss_str.split("_")[1:]
        tcount = int(elems[0][:-1])
        size = int(elems[1][:-1])
        difficult = elems[2] == "d"
    else:
        raise ValueError(f"Unknown bss mode [{bss_str[0]}]")
    return tcount,size,difficult

def get_block_search_space(cfg):
    tcount,size,difficult = parse_bss(cfg.bss_str)
    bss = get_cached_block_search_space(cfg.bss_dir,cfg.nblocks,cfg.nframes,tcount,size,difficult)
    loader = get_bss_batch_loader(cfg,bss)
    return bss,loader

def get_bss_batch_loader(cfg,bss):
    data = edict()
    data.te = WrapperBSSDataset(bss)
    data.tr,data.val = data.te,data.te
    drop_last = cfg.drop_last['te']
    cfg.drop_last['te'] = False
    loader = get_loader(cfg,data,cfg.bss_batch_size,None)
    cfg.drop_last['te'] = drop_last
    return loader.te

def get_cached_block_search_space(bss_dir,nblocks,nframes,tcount,size,difficult=False):
    if not bss_dir.exists(): bss_dir.mkdir()
    bss_fn = bss_dir / f"block_arange_{nblocks}b_{nframes}f_{tcount}t_{size}s.npy"
    REF_H = get_ref_block_index(nblocks)
    if False and bss_fn.exists():
        if verbose: print(f"[Reading bss]: {bss_fn}")
        block_search_space = np.load(bss_fn,allow_pickle=True)
        bss = block_search_space
    else:
        block_search_space = generate_block_search_space(nblocks,nframes,tcount,difficult)
        bss = block_search_space
        if verbose: print(f"Original block search space: [{len(bss)}]")
        if len(bss) >= size:
            # bss = bss_subsample(bss,size,REF_H,nframes,"random")
            # bss = bss_subsample(bss,size,REF_H,nframes,"along_frame")
            bss = bss_subsample(bss,size,REF_H,nframes,"almost_nodynamics")
            block_search_space = bss
        bss = block_search_space
        if verbose: print(f"Writing block search space: [{bss_fn}]")
        if not isinstance(bss,torch.Tensor): bss = torch.stack(bss)
        bss = bss.numpy()
        np.save(bss_fn,bss)
        bss = torch.LongTensor(bss)
    if verbose: print(f"Search Space Size: {len(block_search_space)}")
    return bss

def bss_subsample(bss,size,REF_H,nframes,method="random"):
    if method == "random":
        bss = remove_nodynamics(bss,REF_H) 
        bss = torch.stack(random.sample(list(bss),size-1),dim=0)
        bss = ensure_single_nodynamics(bss,REF_H,nframes)
        return bss
    elif method == "along_frame":
        bss = remove_nodynamics(bss,REF_H) 
        bss = bss[:size-1]
        bss = ensure_single_nodynamics(bss,REF_H,nframes)
        return bss
    elif method == "almost_nodynamics":
        bss = remove_nodynamics(bss,REF_H) 
        BSS = bss.shape[0]
        deltas = torch.sum(torch.abs(bss - REF_H),1)        
        args = torch.nonzero(deltas < (nframes//2-1) )[:,0]
        if len(args) < size:
            n = size - len(args) - 1
            set_all = set(list(np.arange(BSS)))
            set_args = set(list(args.numpy()))
            remaining = set_all - set_args
            remaining = torch.LongTensor(list(remaining))
            remaining = remaining[:n]
            index = torch.cat([args,remaining],dim=0)
            bss = bss[index]
        else:
            bss = bss[args[:size-1]]
        bss = ensure_single_nodynamics(bss,REF_H,nframes)
        return bss
    else:
        raise ValueError(f"Uknown bss_subsample method [{method}]")

def generate_block_search_space(nblocks,nframes,tcount=10,difficult=False):
    # -- create grid over neighboring blocks --
    nh_grid = np.arange(nblocks**2)
    REF_H = get_ref_block_index(nblocks)
    
    # -- rand for nframes-1 --
    nh_grid_rep = []
    for t in range(nframes-1):
        diff_valid = tcount < nblocks**2
        if difficult and diff_valid:
            mid_point = tcount//2
            rand = []
            for t_index in range(tcount):
                # -- skip correct alignment --
                rand.append(REF_H-mid_point+t_index) 
            rand = np.array(rand)
        else:
            rand = np.random.permutation(len(nh_grid))[:tcount]
        nh_grid_rep.append(rand)
    grids = np.meshgrid(*nh_grid_rep,copy=False)
    grids = np.array([grid.ravel() for grid in grids]).T

    # -- fix index for reference patch --
    ref_index = REF_H * torch.ones(grids.shape[0])
    M = ( grids.shape[1] ) // 2
    grids = np.c_[grids[:,:M],ref_index,grids[:,M:]]

    # -- convert for torch --
    grids = torch.LongTensor(grids)
    grids = torch.flip(grids,(1,))

    # -- single dynamics middle block --
    # grids = remove_nodynamics(grids,REF_H)
    grids = ensure_single_nodynamics(grids,REF_H,nframes)

    return grids


def args_nodynamics_nblocks(grids,nblocks):
    REF_H = get_ref_block_index(nblocks)
    deltas = torch.sum(torch.abs(grids - REF_H),1)
    args = torch.nonzero(deltas == 0)
    return args

def args_nodynamics(grids,REF_H):
    deltas = torch.sum(torch.abs(grids - REF_H),1)
    args = torch.nonzero(deltas == 0)
    return args

def ensure_single_nodynamics(grids,REF_H,nframes):
    args = args_nodynamics(grids,REF_H)
    if len(args) == 0:
        nodynamics = torch.tensor(np.array([REF_H]*nframes)).long()[None,:]
        grids = torch.cat([nodynamics,grids],dim=0)
    return grids

def remove_nodynamics(grids,REF_H):
    args = args_nodynamics(grids,REF_H)
    if len(args) > 0:
        args = args[0].item()
        grids = torch.cat([grids[:args],grids[args+1:]],dim=0)
    return grids

class WrapperBSSDataset():
    def __init__(self,bss):
        self.bss = torch.LongTensor(bss)

    def __len__(self):
        return len(self.bss)

    def __getitem__(self,index):
        order = self.bss[index]
        return {'order':order}
