
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
from pyutils import save_image,images_to_psnrs
from patch_search import get_score_function

# -- local project imports --
from .utils import get_ref_block_index
from .bss import get_block_search_space,args_nodynamics_nblocks
from .blocks import create_image_volumes,create_image_tiles
from .wrap_image_data import load_image_dataset,sample_to_cuda
# from .exps import run_fnet,run_pixel_scores,run_cog,run_pixel_diffs,run_flownet
from .exps import PixelExperiment,FlownetExperiment

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

    # -- setup experiments --
    # image_exps = [FlownetExperiment]
    image_exps = []
    block_exps = [PixelExperiment]
    exp_dict = edict({'image':image_exps,'block':block_exps})
    exps = setup_experiments(cfg,exp_dict)

    # -- 1.) over BATCHES of IMAGES  --
    # results = {'image_bindex':[],'bss_bindex':[]}
    results = {'bss':[],'bss_ibatch':[]}
    NUM_BATCHES = 2
    for image_bindex in tqdm(range(NUM_BATCHES),leave=False):

        # -- init results --
        results_i = {}

        # -- sample & unpack batch --
        sample = next(image_batch_iter)
        sample_to_cuda(sample)
        dyn_noisy = sample['noisy'] # dynamics and noise
        dyn_clean = sample['burst'] # dynamics and no noise
        static_noisy = sample['snoisy'] # no dynamics and noise
        static_clean = sample['sburst'] # no dynamics and no noise
        flow = sample['flow']
        
        # T = cfg.nframes
        # for t in range(T):
        #     psnrs = images_to_psnrs(dyn_clean[t],dyn_clean[T//2])
        #     print(t,np.mean(psnrs))

        # -- create image volumes --
        T,B,C,H,W = static_clean.shape
        static_vols = create_image_volumes(cfg,static_clean,static_noisy)
        dyn_vols = create_image_volumes(cfg,dyn_clean,dyn_noisy)
        T,H2,B,P,C,PS,PS = static_vols.clean.shape
        image_batch_size = B

        # -- run experiments over just image batch
        # tiles = create_image_tiles(static_vols.clean,static_vols.noisy,flow,H)
        # tiles = create_image_tiles(dyn_vols.clean,dyn_vols.noisy,flow,H)
        # exp_results = execute_experiments(cfg,exps.image,tiles,flow)
        # full = edict({'clean':dyn_clean,'noisy':dyn_noisy})
        # exp_results = execute_experiments(cfg,exps.image,full,flow)
        # format_tensor_results(cfg,exp_results,results_i,{'default':-1},True)

        # -- restart bss loader --
        bss_iter = iter(bss_loader)

        # -- 2.) over BATCHES of BLOCK_ORDERs  --
        tgrid = torch.arange(cfg.nframes)
        for block_bindex in tqdm(range(BSS_SIZE),leave=False):

            # -- sample block order --
            blocks = next(bss_iter)['order']

            # -- get image regions with no dynamics --
            args = args_nodynamics_nblocks(blocks,cfg.nblocks)
            print(args)

            # -- pick block order (aka arangement) --
            shape_str = 'e t b p c h w -> p b e t c h w'
            clean = rearrange(static_vols.clean[tgrid,blocks],shape_str)
            noisy = rearrange(static_vols.noisy[tgrid,blocks],shape_str)
            P,B,E,T,C,PS,PS = clean.shape # explicit shape

            # -- cuda -- 
            clean = clean.to(cfg.device)
            noisy = noisy.to(cfg.device)

            # -- run experiment suit --
            block_batch = edict({'clean':clean,'noisy':noisy})
            exp_results = execute_experiments(cfg,exps.block,block_batch,flow)

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

def setup_experiments(cfg,exp_dict):
    init_exps = {}
    for type_name,exp_list in exp_dict.items():
        init_exps[type_name] = []
        for exp in exp_list:
            init_exps[type_name].append(exp(cfg))
    init_exps = edict(init_exps)
    return init_exps
        
def execute_experiments(cfg,exps,batch,flow):
    results = {}
    for exp in exps:
        exp.run(cfg,batch.clean,batch.noisy,flow,results)
    print(list(results.keys()))
    return results

# def execute_block_batch_experiments(cfg,exps,block_batch,flow):
#     results = {}
#     # run_fnet(cfg,clean,noisy,directions,results)
#     run_pixel_scores(cfg,block_batch.clean,block_batch.noisy,flow,results)
#     # run_cog(cfg,clean,noisy,directions,results)
#     # run_pixel_diffs(cfg,clean,noisy,directions,results)
#     return results


