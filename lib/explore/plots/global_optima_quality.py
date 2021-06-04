"""
Assess the quality of the 
global optima using a specific score
function 
by comparing with
the true global optima (no dynamics)

"""

# -- python imports --
import numpy as np
import numpy.random as npr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange
from easydict import EasyDict as edict
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# -- pytorch imports --
import torch
import torch.nn.functional as f

# -- explore imports --
from explore.plots.utils import move_column_to_last,get_pixel_fields,find_optima_index


def compare_global_optima_quality(cfg,records,exp_fields):
    quality = []
    for exp_int in range(len(records)):
        quality_i = experiment_global_optima_quality(cfg,records,
                                                     exp_fields,exp_int)
        
        # -- append extra info
        quality_df = pd.DataFrame(quality_i).T
        record = records[exp_int]
        fields = ['nframes',
                  'nblocks',
                  'noise_type',
                  'patchsize',
                  'random_seed']
        for field in fields:
            quality_df[field] = record[field]

        # -- format pandas results --
        quality_df.reset_index(inplace=True)
        quality_df = quality_df.rename(columns={'index':'metric'})
        quality.append(quality_df)

    quality = pd.concat(quality)
    quality = move_column_to_last(quality,"random_seed")
    quality = move_column_to_last(quality,"correct_v")

    print(quality)
    fields = get_pixel_fields(records[0])
    for field in fields:
        quality_metric = quality[quality['metric'] == field]
        print(field,quality_metric['acc'].mean())
    return quality


def experiment_global_optima_quality(cfg,records,exp_fields,exp_int):

    # -- extract & print summary info --
    record = records[exp_int]
    for exp_field in exp_fields:
        if exp_field == 'bss': continue
        print(exp_field,record[exp_field])
    bss = record['bss']
    bss_ibatch = record['bss_ibatch']
    nblocks = record['nblocks']
    optima_index = find_optima_index(bss,bss_ibatch,nblocks)
    # nodynamic_index,optima_nd_index = find_nodynamic_optima_idx(bss,bss_ibatch,nblocks)

    # -- compute optima quality for each pixel function --
    fields = get_pixel_fields(record)
    quality = edict({field: edict({'correct':0,'total':0,'acc':0}) for field in fields})
    for field in fields:
        scores = rearrange(records[exp_int][field]['scores'],'p ib ss -> ss ib p')
        scores_t = rearrange(records[exp_int][field]['scores_t'],'p ib ss t -> ss t ib p')
        batch_size = records[exp_int]['batch_size']
        SS,IB,P = scores.shape
        # search_space batch, image batch, npatches
        scores = torch.mean(scores,dim=2) # ave over image patches
        
        # -- find optimal score index --
        argmins = torch.argmin(scores,0)

        # -- compute quality --
        quality[field].correct = torch.sum(optima_index == argmins).item()
        quality[field].total = len(optima_index)
        quality[field].acc = 100*quality[field].correct/quality[field].total
        quality[field].correct_v = '.'.join([str(x) for x in list((optima_index == argmins).numpy().astype(int))])

        # # -- filter bss to only almost no dynamics --
        # scores = scores[nodynamic_index]
        # argmins = torch.argmin(scores,0)

        # # -- compute quality --
        # quality[field].nd_correct = torch.sum(optima_nd_index == argmins).item()
        # quality[field].nd_total = len(optima_index)
        # quality[field].nd_acc = 100*quality[field].nd_correct/quality[field].nd_total

    return quality


        
