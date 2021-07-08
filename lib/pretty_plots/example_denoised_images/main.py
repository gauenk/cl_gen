# -- python imports --
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from pathlib import Path
from collections import OrderedDict
from einops import rearrange
import matplotlib as mpl
import matplotlib.pyplot as plt
import textwrap

# -- pytorch imports --
import torch
import torchvision.utils as tv_utils
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils import add_legend,save_image,print_tensor_stats
from lpas.main import get_main_config
from datasets.transforms import get_noise_config,set_noise,get_noise_transform
from explore.wrap_image_data import load_image_dataset,sample_to_cuda

def update_noise_cfg(cfg,noise_type):
    nconfig = get_noise_config(cfg,noise_type)
    set_noise(cfg,nconfig[nconfig.ntype])
    # cfg.noise_type = nconfig.ntype
    # cfg.noise_params = nconfig
    return cfg

def get_noise_xform(cfg,noise_type):
    cfg = update_noise_cfg(cfg,noise_type)
    noise_xform = get_noise_transform(cfg.noise_params,use_to_tensor=False)
    return noise_xform

def simulate_noisy_images(cfg,clean,noises):
    noisy_l = []
    for noise_type in noises:
        noise_xform = get_noise_xform(cfg,noise_type)
        noisy = noise_xform(clean)+0.5
        noisy_l.append(noisy)
    return noisy_l

def save_noise_model_misspecification(cfg,clean):

    # -- save a seq of noisy images --
    noise_types = ["pn-4p0-0p0","g-75p0"]
    noise_seq = simulate_noisy_images(cfg,clean,noise_types)
    DIR = Path("./output/pretty_plots")
    if not DIR.exists(): DIR.mkdir()
    fn = DIR / "image_noise_sequence.png"
    tv_utils.save_image(noise_seq,fn,normalize=True)
    print(f"Wrote image file to [{fn}]")

    # -- save a seq of noisy images --
    noise_types = ["pn-4p0-0p0","pn-4p0-0p0"]
    noise_seq = simulate_noisy_images(cfg,clean,noise_types)
    DIR = Path("./output/pretty_plots")
    if not DIR.exists(): DIR.mkdir()
    fn = DIR / "image_noise_pair_pn-pn.png"
    tv_utils.save_image(noise_seq,fn,normalize=True)
    print(f"Wrote image file to [{fn}]")

    # -- save a seq of noisy images --
    noise_types = ["pn-4p0-0p0","clean"]
    noise_seq = simulate_noisy_images(cfg,clean,noise_types)
    DIR = Path("./output/pretty_plots")
    if not DIR.exists(): DIR.mkdir()
    fn = DIR / "image_noise_pair_pn-clean.png"
    tv_utils.save_image(noise_seq,fn,normalize=True,scale_each=True)
    print(f"Wrote image file to [{fn}]")



def run():

    # -- get experiment config --
    cfg = get_main_config()
    cfg.batch_size = 100
    cfg.nframes = 3
    cfg.frame_size = 350
    cfg.N = cfg.nframes
    cfg.dynamic.frame_size = cfg.frame_size
    cfg.dynamic.frames = cfg.nframes
    cfg.gpuid = 0
    cfg.random_seed = 0    
    T = cfg.nframes

    # -- setup seed --
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # -- load image data --
    data,loader = load_image_dataset(cfg)
    data_iter = iter(loader.tr)

    # -- get image sample --
    N = 2
    for i in range(N):
        sample = next(data_iter)
    dyn_noisy = sample['noisy'] # dynamics and noise
    dyn_clean = sample['burst'] # dynamics and no noise
    static_noisy = sample['snoisy'] # no dynamics and noise
    static_clean = sample['sburst'] # no dynamics and no noise
    flow = sample['flow']
    # save_image(dyn_clean[T//2],"samples.png")
    pick = 26
    save_image(dyn_clean[T//2,pick],"samples.png")
    cropped = True

    # -- get picks --
    noisy = dyn_noisy[T//2,pick]+0.5
    clean = dyn_clean[T//2,pick]+0.5

    # -- optionally crop --
    if cropped:
        noisy = tvF.crop(noisy,150,0,275,125)
        clean = tvF.crop(clean,150,0,275,125)
        
    # -- noise model misspecification --
    save_noise_model_misspecification(cfg,clean)

    # -- pic list --
    pics = OrderedDict()
    psnrs = OrderedDict()

    # -- get clean ref --
    pics['Clean'] = clean
    psnrs['Clean'] = None

    # -- apply noise to image --
    noise_type = "g-100p0"
    noise_xform = get_noise_xform(cfg,noise_type)
    noisy = noise_xform(clean)+0.5
    pics['Noisy'] = noisy
    psnrs['Noisy'] = None

    # -- RAFT --
    noise_type = "g-25p0"
    noise_xform = get_noise_xform(cfg,noise_type)
    raft = noise_xform(clean)+0.5
    pics['RAFT'] = raft
    psnrs['RAFT'] = 25.

    # -- LSRMTF --
    noise_type = "g-30p0"
    noise_xform = get_noise_xform(cfg,noise_type)
    lsrmtf = noise_xform(clean)+0.5
    # pics['lsrmtf'] = lsrmtf

    # -- NNF --
    noise_type = "g-10p0"
    noise_xform = get_noise_xform(cfg,noise_type)
    nnf = noise_xform(clean)+0.5
    pics['LDOF'] = nnf
    psnrs['LDOF'] = 29.

    # -- OURS --
    noise_type = "g-5p0"
    noise_xform = get_noise_xform(cfg,noise_type)
    ours = noise_xform(clean)+0.5
    pics['Ours'] = ours
    psnrs['Ours'] = 32.

    # -- OURS+SEG --
    noise_type = "g-2p0"
    noise_xform = get_noise_xform(cfg,noise_type)
    ours_seg = noise_xform(clean)+0.5
    pics['Ours+Seg'] = ours_seg
    psnrs['Ours+Seg'] = 36.
    
    # -- ALL PICS --
    names = list(pics.keys())
    M = len(names)
    dpi = mpl.rcParams['figure.dpi']
    H,W = clean.shape[-2:]
    figsize = M * (W / float(dpi)), H / float(dpi)
    fig,axes = plt.subplots(1,M,figsize=figsize,sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.05,wspace=0.05)
    for i,ax in enumerate(axes):
        name = names[i]
        pic = rearrange(pics[name],'c h w -> h w c')
        pic = torch.clip(pic,0,1)
        ax.imshow(pic,interpolation='none',aspect='auto')
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # label = textwrap.fill(name + '\n' + '1.2',15)
        label = name
        if not(psnrs[name] is None): label += '\n' + str(psnrs[name])
        ax.set_xlabel(label,fontsize=12)


    # -- save plot --
    DIR = Path("./output/pretty_plots")
    if not DIR.exists(): DIR.mkdir()
    if cropped: fn =  DIR / "./example_denoised_images_cropped.png"
    else: fn =  DIR / "./example_denoised_images.png"
    plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")

