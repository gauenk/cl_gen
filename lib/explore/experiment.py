
# -- python imports --
import numpy as np
import pandas as pd
from tqdm import tqdm
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.transforms as tvT

# -- project imports --
from pyutils import save_image
from patch_search import get_score_function

# -- local project imports --
from .utils import get_ref_block_index
from .bss import get_block_search_space,args_nodynamics_nblocks
from .blocks import create_image_volumes
from .wrap_image_data import load_image_dataset
from .exps import run_fnet,run_pixel_scores,run_cog,run_pixel_diffs

def execute_experiment(cfg):

    # -- set random seed --
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # -- set field cache dir --
    cfg.field_cache_dir = cfg.explore_dir / cfg.uuid

    # -- load data --
    image_data,image_batch_loaders = load_image_dataset(cfg)
    image_batch_iter = iter(image_batch_loaders.tr)

    # -- get block search space --
    bss_data,bss_loader = get_block_search_space(cfg)
    bss_iter = iter(bss_loader)
    BSS_SIZE = len(bss_loader)
    
    # REF_H = get_ref_block_index(cfg.nblocks)
    # # print(bss_data.shape)
    # deltas = torch.sum(torch.abs(bss_data - REF_H),1)
    # print(torch.sum(deltas < (cfg.nframes//2-1) ))
    # exit()


    # -- 1.) over BATCHES of IMAGES  --
    # results = {'image_bindex':[],'bss_bindex':[]}
    results = {'bss':[],'bss_ibatch':[]}
    NUM_BATCHES = 2
    for image_bindex in tqdm(range(NUM_BATCHES),leave=False):

        # -- restart bss loader --
        bss_iter = iter(bss_loader)

        # -- sample batch --
        loaded = next(image_batch_iter)
        clean_imgs,noisy_imgs = loaded['burst'],loaded['noisy']
        directions = loaded['directions']

        # -- create image volumes --
        T,B,C,H,W = clean_imgs.shape
        clean_vol,noisy_vol = create_image_volumes(cfg,clean_imgs,noisy_imgs)
        T,H2,B,P,C,PS,PS = clean_vol.shape
        image_batch_size = B

        # -- 2.) over BATCHES of BLOCK_ORDERs  --
        results_i = {}
        tgrid = torch.arange(T)
        for block_bindex in tqdm(range(BSS_SIZE),leave=False):

            # -- sample block order --
            blocks = next(bss_iter)['order']
            # print("blocks",blocks[:10])
            # print(blocks[70])
            args = args_nodynamics_nblocks(blocks,cfg.nblocks)
            print(args)

            # -- pick block order (aka arangement) --
            clean = rearrange(clean_vol[tgrid,blocks],'e t b p c h w -> p b e t c h w')
            noisy = rearrange(noisy_vol[tgrid,blocks],'e t b p c h w -> p b e t c h w')
            P,B,E,T,C,PS,PS = clean.shape # explicit shape

            # -- cuda -- 
            clean = clean.to(cfg.device)
            noisy = noisy.to(cfg.device)

            # -- run experiment suit --
            exp_results = execute_batch_experiments(cfg,clean,noisy,directions)

            # -- store results --
            format_tensor_results(cfg,exp_results,results_i,{'default':-1},True)
            # -- block search space is constant --
            blocks = repeat(blocks,'bss_bs t -> img_bs bss_bs t',img_bs=image_batch_size)
            format_tensor_results(cfg,{'bss':[blocks],'bss_ibatch':[th([B])]},
                                  results_i,{'default':-1},True)
            # print(results_i)

        # -- list of vectors to torch tensor --
        dims = {'bss':1,'bss_ibatch':0,'default':2}
        format_tensor_results(cfg,results_i,results,dims,append=True)
        # print(results['pixel_jackknife'].shape)
        # print(list(results.keys()))
        

    # -- convert ndarrays into files --
    dims = {'bss':0,'bss_ibatch':0,'default':1}
    format_tensor_results(cfg,results,results,dims,append=False)
    print_tensor_fields(results)

    return results

def th(int_number):
    return torch.LongTensor(int_number)

def print_tensor_fields(results):
    for fieldname,results_f in results.items():
        print(fieldname,type(results_f))
        if not isinstance(results_f,dict):
            print(results_f.shape)
            continue
        for sname,results_s in results_f.items():
            print(sname,results_s.shape)

def format_tensor_results(cfg,results_i,results_o,dims,append=True):
    """
    results_i (dict): results input
    { "pixel_diff": {"scores":[...],"scores_t":[...],...},
      "cog_v1": {"scores":[...],"scores_t":[...],...},
      "bss": [...],
      ... }
    """
    for metric_group in results_i.keys():
        # -- select dimension --
        if metric_group in dims.keys(): dim = dims[metric_group]
        else: dim = dims['default']

        # -- select format func --
        mgroup = results_i[metric_group]
        if isinstance(mgroup,dict): 
            format_tensor_dict(cfg,metric_group,results_i,results_o,dim,append)
        elif isinstance(mgroup,list): 
            format_tensor_list(cfg,metric_group,results_i,results_o,dim,append)
        else:
            raise TypeError(f"Uknown metric group type [{type(mgroup)}]")

def format_tensor_list(cfg,metric_group,results_i,results_o,dim,append=True):
    # -- Note: metric is a misnomer in this function --

    # -- init metric group if not already --
    if not(metric_group in results_o.keys()): results_o[metric_group] = []

    # -- group together into output result --
    metric = results_i[metric_group]
    # -- cat together potential list --
    if isinstance(metric,list): metric = torch.cat(metric,dim=dim)
    # -- append result if necessary --
    if append: results_o[metric_group].append(metric)
    else: results_o[metric_group] = metric

def format_tensor_dict(cfg,metric_group,results_i,results_o,dim,append=True):

    # -- init metric group if not already --
    if not(metric_group in results_o.keys()):
        results_o[metric_group] = {}
        for metric_name in results_i[metric_group].keys():
            results_o[metric_group][metric_name] = []

    # -- group together into output result --
    for metric_name in results_i[metric_group].keys():
        metric = results_i[metric_group][metric_name]
        # -- cat together potential list --
        if isinstance(metric,list): metric = torch.cat(metric,dim=dim)
        # -- append result if necessary --
        if append: results_o[metric_group][metric_name].append(metric)
        else: results_o[metric_group][metric_name] = metric

def execute_batch_experiments(cfg,clean,noisy,directions):
    results = {}
    # run_fnet(cfg,clean,noisy,directions,results)
    run_pixel_scores(cfg,clean,noisy,directions,results)
    # run_cog(cfg,clean,noisy,directions,results)
    # run_pixel_diffs(cfg,clean,noisy,directions,results)
    return results


