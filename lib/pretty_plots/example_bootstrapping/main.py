# -- python imports --
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from pathlib import Path
from collections import OrderedDict
from einops import rearrange,repeat
import matplotlib as mpl
import matplotlib.pyplot as plt
import textwrap

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF
import torchvision.utils as tv_utils

# -- project imports --
from settings import ROOT_PATH
from pyutils import add_legend,save_image,print_tensor_stats,sample_subset_grids
from lpas.main import get_main_config
from datasets.transforms import get_noise_config,get_noise_transform
from explore.wrap_image_data import load_image_dataset,sample_to_cuda
from explore.bss import get_block_search_space,args_nodynamics_nblocks
from explore.blocks import create_image_volumes,create_image_tiles


# -- global vars --

DIR = Path(f"{ROOT_PATH}/output/pretty_plots/")

#
# GATHER DATA
#

def get_cfg():
    cfg = get_main_config()
    cfg.batch_size = 100
    cfg.nframes = 3
    cfg.frame_size = 350
    cfg.N = cfg.nframes
    cfg.dynamic.frame_size = cfg.frame_size
    cfg.dynamic.frames = cfg.nframes
    cfg.gpuid = 0
    cfg.random_seed = 0    
    cfg.bss_str = '0m_2f_200t_d'
    cfg.bss_batch_size = 100
    cfg.npatches = 6
    cfg.patchsize = 3
    cfg.nblocks = 3
    cfg.nh_size = cfg.nblocks # old name
    cfg.explore_package = "lpas" # -- pick package to explore --
    cfg.explore_dir = Path(ROOT_PATH) / f"./output/explore/{cfg.explore_package}/"
    if not cfg.explore_dir.exists(): cfg.explore_dir.mkdir(parents=True)
    cfg.bss_dir = cfg.explore_dir / "./bss/"
    if not cfg.bss_dir.exists(): cfg.bss_dir.mkdir(parents=True)
    cfg.drop_last = {'tr':True,'val':True,'te':True}

    # -- set noise params --
    nconfig = get_noise_config(cfg,"g-75p0")
    cfg.noise_type = nconfig.ntype
    cfg.noise_params = nconfig


    return cfg

def get_bootstrapping_results_to_plot():

    # -- get experiment config --
    cfg = get_cfg()

    # -- setup seed --
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # -- load data --
    image_data,image_batch_loaders = load_image_dataset(cfg)
    image_batch_iter = iter(image_batch_loaders.tr)

    # -- get block search space --
    bss_data,bss_loader = get_block_search_space(cfg)
    bss_iter = iter(bss_loader)
    BSS_SIZE = len(bss_loader)
    bss_iter = iter(bss_loader)

    # -- sample & unpack batch --
    sample = next(image_batch_iter)
    sample_to_cuda(sample)
    dyn_noisy = sample['noisy'] # dynamics and noise
    dyn_clean = sample['burst'] # dynamics and no noise
    static_noisy = sample['snoisy'] # no dynamics and noise
    static_clean = sample['sburst'] # no dynamics and no noise
    flow = sample['flow']

    # -- create vols --
    static_vols = create_image_volumes(cfg,static_clean,static_noisy)
    dyn_vols = create_image_volumes(cfg,dyn_clean,dyn_noisy)

    tgrid = torch.arange(cfg.nframes)
    for block_bindex in tqdm(range(BSS_SIZE),leave=False):
        
        # -- sample block order --
        blocks = next(bss_iter)['order']

        # -- get image regions with no dynamics --
        args = args_nodynamics_nblocks(blocks,cfg.nblocks)
        print(args)
        if len(args) == 0: continue
        
        # -- pick block order (aka arangement) --
        shape_str = 'e t b p c h w -> p b e t c h w'
        clean = rearrange(static_vols.clean[tgrid,blocks],shape_str)
        noisy = rearrange(static_vols.noisy[tgrid,blocks],shape_str)
        P,B,E,T,C,PS,PS = clean.shape # explicit shape
        
        # -- compute pretty plots bootstrapping 
        bootstrapping_results_to_plot = pretty_plots_bootstrapping(cfg,noisy)
        bootstrapping_results_to_plot['optima'] = args
        break
    return bootstrapping_results_to_plot

def pretty_plots_bootstrapping(cfg,expanded):

    # -- setup vars --
    R,B,E,T,C,H,W = expanded.shape
    device = expanded.device
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    samples = samples.contiguous() # speed up?
    t_ref = T//2
    ave = torch.mean(samples,dim=0)

    # -- compute ave diff between model and subsets --
    subset_examples = []
    scores_t = torch.zeros(T,B*E,device=device)
    counts_t = torch.zeros(T,1,device=device)
    nbatches,batch_size = 1,100
    for batch_idx in range(nbatches):
        subsets = torch.LongTensor(sample_subset_grids(T,batch_size))
        subset_examples.append(subsets)
        for subset in subsets:
            counts_t[subset] += 1
            subset_pix = samples[subset]
            subset_ave = torch.mean(subset_pix,dim=0)
            loss = torch.mean( (subset_ave - ave)**2, dim=0)
            scores_t[subset] += loss
    scores_t /= counts_t
    scores = torch.mean(scores_t,dim=0)
    scores_t = scores_t.T # (T,E) -> (E,T)

    # -- no cuda --
    scores = rearrange(scores.cpu(),'(b e) -> b e',b=B)
    scores_t = rearrange(scores_t.cpu(),'(b e) t -> b e t',b=B)

    # -- add back patchsize for compat --
    scores = repeat(scores,'b e -> r b e',r=R)
    scores_t = repeat(scores_t,'b e t -> r b e t',r=R)

    # -- create info --
    info = {'subsets':subset_examples[0],
            'expanded':expanded,
            'ave':ave,
            'counts_t':counts_t,
            'scores_t':scores_t,
            'scores':scores}

    return info

#
# PLOTTING
#

def get_sample_bootstrapping_info(expanded,subsets,ave):
    # -- shapes --
    R,B,E,T,C,H,W = expanded.shape
    samples = rearrange(expanded,'r b e t c h w -> t (r c h w) (b e)')
    infos = []
    for subset in subsets:

        # -- compute subset for boostrap --
        subset_pix = samples[subset]
        subset_ave = torch.mean(subset_pix,dim=0)
        # print(subset_pix.shape)
        # print(subset_ave.shape)
        # print(ave.shape)
        loss = torch.mean( (subset_ave - ave)**2, dim=0)

        # -- reshaping -- 
        shape_str = 't (r c h w) (b e) -> r b e t c h w'
        subset_pix = rearrange(subset_pix,shape_str,c=C,h=H,w=W,b=B)

        shape_str = '(r c h w) (b e) -> r b e c h w'
        subset_ave = rearrange(subset_ave,shape_str,c=C,h=H,w=W,b=B)

        shape_str = '(r c h w) (b e) -> r b e c h w'
        ave_rs = rearrange(ave,shape_str,c=C,h=H,w=W,b=B)
        
        shape_str = '(b e) -> b e'
        loss = rearrange(loss,shape_str,b=B)

        # -- save --
        info = {'subset':subset,
                'subset_pix':subset_pix,
                'subset_ave':subset_ave,
                'ave':ave_rs,
                'loss':loss}
        infos.append(info)
    return infos

def format_results_to_plot(results):

    # -- unpack pixels --
    expanded = results['expanded']
    subsets = results['subsets']
    ave = results['ave']
    optima = results['optima']
    
    # -- show non optima --
    subsets = torch.cat([subsets[-1:],subsets[:1],subsets[[3]]],dim=0)
    infos = get_sample_bootstrapping_info(expanded,subsets,ave)

    return infos 

def plot_examples_bootstrapping():
    # -- get info for plotting --
    results = get_bootstrapping_results_to_plot()
    infos = format_results_to_plot(results)
    optima = results['optima'].item()
    print("Optima at index %d" % optima)
    create_bootstrapping_plot(results,infos,0,0)
    create_bootstrapping_plot(results,infos,0,1)
    create_bootstrapping_plot(results,infos,0,optima)
    
def create_bootstrapping_plot(results,infos,b_index,e_index):

    # -- unpack pixels --
    expanded = results['expanded']
    ave = infos[0]['ave']
    R,B,E,T,C,H,W = expanded.shape

    # -- create an image bar --
    img_bars,labels = [],[]
    img_bar = create_image_bar(np.arange(T),b_index,e_index,expanded,ave)
    img_bars.append(img_bar)
    labels.append("Ref")
    for info in infos:
        subset = info['subset']
        print(subset)
        subset_ave = info['subset_ave']
        score = info['loss'][b_index,e_index].item()
        score_str = "%2.3e" % score
        img_bar = create_image_bar(subset,b_index,e_index,expanded,subset_ave)
        img_bars.append(img_bar)
        labels.append(score_str)
        
    # -- plot image bars --
    pad = 2
    dpi = 300
    M = T+1
    L = len(img_bars)
    figsize = (M*4,L*4)
    # figsize = M * (W + pad ) / float(dpi), (H+pad) / float(dpi)
    # figsize = M * (W / float(dpi)), H / float(dpi)
    # fig,axes = plt.subplots(1,M,figsize=figsize,sharex=True,sharey=True)
    fig,axes = plt.subplots(L,1,figsize=figsize,sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.05,wspace=0.05)
    idx = 0
    for img_bar,label,ax in zip(img_bars,labels,axes):
        img_bar_2plot = rearrange(img_bar,'c h w -> h w c')
        img_bar_2plot += 0.5
        img_bar_2plot = torch.clip(img_bar_2plot,0,1)
        # print_tensor_stats("img_bar",img_bar_2plot)
        ax.imshow(img_bar_2plot)
        # ax.text(1.1, 1.1, f'hi_{idx}',fontsize=20)
        idx += 1
        
    # -- save figure --
    if not DIR.exists(): DIR.mkdir()
    fn =  DIR / f"./example_bootstrapping_{b_index}_{e_index}.png"
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0) 
    plt.savefig(fn,transparent=True,dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")
        
        
def create_image_bar(tgrid,b_index,e_index,samples,ave):
    # -- unpack and rename --
    R,B,E,T,C,H,W = samples.shape
    samples = samples[0,b_index,e_index]
    ave = ave[0,b_index,e_index]

    # -- extract images in correct locations --
    imgs = []
    for t in range(T):
        if t in tgrid: img = samples[t].cpu()
        else: img = torch.zeros(C,H,W).cpu()
        imgs.append(img)
    imgs.append(ave.cpu())
    
    # -- create image grid --
    grid = tv_utils.make_grid(imgs,nrow=T+1)

    return grid
    

def run():
    plot_examples_bootstrapping()


