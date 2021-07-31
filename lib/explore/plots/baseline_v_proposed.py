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
from explore.utils import get_ref_block_index
from explore.bss import get_block_search_space,args_nodynamics_nblocks
from explore.blocks import create_image_volumes
from explore.wrap_image_data import load_image_dataset
# from explore.exps import run_fnet,run_pixel_scores,run_cog,run_pixel_diffs


def compare_baseline_v_proposed(cfg,cfg_setup,records,exp_fields):

    # -- collect results from experiments --
    info = []
    fields = ['nframes','nblocks','noise_type','patchsize','random_seed']
    for exp_int in range(len(records)):
        info_i = get_experiment_info(cfg,cfg_setup,records,exp_fields,exp_int)
        # -- append extra info
        info_df = pd.DataFrame(info_i).T
        record = records[exp_int]
        for field in fields:
            quality_df[field] = record[field]

        # -- format pandas results --
        info_df.reset_index(inplace=True)
        info_df = info_df.rename(columns={'index':'metric'})
        info.append(info_df)
    info = pd.concat(info)
    print(info)
        
        
def get_experiment_info(cfg,cfg_setup,records,exp_fields,exp_int):

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
    
    #
    # Setup Analysis
    # 

    # -- set random seed --
    config = edict(records[exp_int])
    cfg = cfg_setup(cfg,config)
    cfg.uuid = config.uuid

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # -- set field cache dir --
    cfg.field_cache_dir = cfg.explore_dir / cfg.uuid

    # -- load data --
    image_data,image_batch_loaders = load_image_dataset(cfg)
    image_batch_iter = iter(image_batch_loaders.tr)

    NUM_BATCHES = 2
    for image_bindex in range(NUM_BATCHES):

        # -- sample batch --
        loaded = next(image_batch_iter)
        clean_imgs,noisy_imgs = loaded['burst'],loaded['noisy']
        directions = loaded['directions']

        # -- create image volumes --
        T,B,C,H,W = clean_imgs.shape
        clean_vol,noisy_vol = create_image_volumes(cfg,clean_imgs,noisy_imgs)
        T,H2,B,P,C,PS,PS = clean_vol.shape
        image_batch_size = B

        # -- compute optima quality for each pixel function --
        fields = get_pixel_fields(record)
        quality = edict({field: edict({'correct':0,'total':0,'acc':0})
                         for field in fields})
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
        
