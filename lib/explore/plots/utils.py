# -- python imports --
import numpy as np
import pandas as pd

# -- pytorch imports --
import torch

# -- project imports --
from explore.utils import get_ref_block_index

def move_column_to_last(df,name):
    columns = df.columns.to_list()
    columns.remove(name)
    columns.append(name)
    df = df.filter(columns)
    return df
    
def get_flownet_fields(exp_records):
    fields,keys = [],list(exp_records.keys())
    for key in keys:
        if "flownet_" in key: fields.append(key)
    return fields
    
def get_pixel_fields(exp_records):
    fields,keys = [],list(exp_records.keys())
    for key in keys:
        if "pixel_" in key: fields.append(key)
    return fields

def find_optima_index(bss,bss_ibatch,nblocks):
    numImageBatches,bssGrid,nframes = bss.shape
    REF_H = get_ref_block_index(nblocks)
    nodynamics = torch.tensor(np.array([REF_H]*nframes)).long()[None,:]
    deltas = torch.sum(torch.abs(torch.tensor(bss) - REF_H),2)
    args = torch.nonzero(deltas == 0)
    args = args[:,1]
    return args

def find_nodynamic_optima_idx(bss,bss_ibatch,nblocks):
    
    numImageBatches,bssGrid,nframes = bss.shape
    REF_H = get_ref_block_index(nblocks)
    nodynamics = torch.tensor(np.array([REF_H]*nframes)).long()[None,:]
    deltas = torch.sum(torch.abs(torch.tensor(bss) - REF_H),2)
    print(bss.shape)
    print(torch.sum(deltas < (nframes//2-1) ))
    args = torch.nonzero(deltas < (nframes//2-1) )
    print(args)
    nodynamic_index = args
    print(nodynamic_index)
    print(bss.shape)

    filtered_bss = bss[nodynamic_index[:,0],nodynamic_index[:,1]]
    print(filtered_bss.shape)
    optima_nd_index = find_optima_index(filtered_bss,bss_ibatch,nblocks)

    return nodynamic_index,optima_nd_index
