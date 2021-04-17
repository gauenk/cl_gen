"""
How fast can we train a unet to denoise a the same
image content with different noise patterns?
"""

# -- python imports --
import sys,os,random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch import --
import torch
from torch import nn
import torchvision.utils as tv_utils
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as tvF

# -- project code --
import settings
from pyutils.timer import Timer
from datasets import load_dataset
from datasets.transforms import get_noise_transform,get_dynamic_transform
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr,count_parameters,images_to_psnrs
from layers.unet import UNet_n2n
from .scores import get_score_function
from .utils import get_block_arangements,get_ref_block_index

FAST_UNET_DIR = Path(f"{settings.ROOT_PATH}/output/lpas/fast_unet/")
if not FAST_UNET_DIR.exists(): FAST_UNET_DIR.mkdir()

def train(cfg,image,clean,burst,model,optim):
    train_steps = 100
    for i in range(train_steps):

        # -- reset --
        model.zero_grad()
        optim.zero_grad()

        # -- rand in and out --
        i,j = random.sample(list(range(burst.shape[0])),2)
        noisy = burst[[i]]
        target = burst[[j]]
        
        # -- forward --
        rec = model(noisy)
        loss = F.mse_loss(rec,target)

        # -- optim step --
        loss.backward()
        optim.step()

def test(cfg,image,clean,burst,model):

    # -- create results --
    results = {}

    # -- repeat along axis --
    rep = repeat(image,'c h w -> tile c h w',tile=burst.shape[0])

    # -- reconstruct a clean image --
    rec = model(burst)+0.5

    # -- compute results --
    loss = F.mse_loss(rec,rep)
    psnr = float(np.mean(images_to_psnrs(rec,rep)))
    results['mse'] = loss.item()
    results['psnr'] = psnr

    # -- compute scores --
    score_fxn_names = ['lgsubset_v_ref','lgsubset','ave','lgsubset_v_indices']
    wrapped_l = []
    for name in score_fxn_names:
        score_fxn = get_score_function(name)
        wrapped_score = score_function_wrapper(score_fxn)
        score = wrapped_score(cfg,rec).item()
        results[name] = score

    # print("Test Loss",loss.item())
    # print("Test PSNR: %2.3e" % np.mean(images_to_psnrs(rec+0.5,rep)))
    tv_utils.save_image(rec,"fast_unet_rec.png",normalize=True)
    tv_utils.save_image(burst,"fast_unet_burst.png",normalize=True)
    return results

def score_function_wrapper(score_fxn):
    def wrapper(cfg,image):
        tmp = image.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scores = score_fxn(cfg,tmp)
        return scores[0,0,0]
    return wrapper

def fast_unet(cfg,data):


    # -- setup noise --
    cfg.noise_type = 'g'
    cfg.noise_params.ntype = cfg.noise_type
    cfg.noise_params['g']['stddev'] = 75. 
    noise_level = 75.
    noise_level_str = f"{int(noise_level)}"
    # nconfig = get_noise_config(cfg,exp.noise_type)
    noise_xform = get_noise_transform(cfg.noise_params,use_to_tensor=False)

    # -- set configs --
    cfg.nframes = 3
    T = cfg.nframes
    cfg.nblocks = 3
    H = cfg.nblocks

    # -- create our neighborhood --
    full_image = data.tr[0][2]

    clean = []
    tl_list = [[128,10],[11,10],[12,10],[12,9],[11,9],[11,8],[12,8],[13,8],[14,8],[14,9]]
    for i in range(-H//2+1,H//2+1):
        for j in range(-H//2+1,H//2+1):
            clean.append(tvF.crop(full_image,128+i,128+j,96,96))
    clean = torch.stack(clean,dim=0)
    REF_H = get_ref_block_index(cfg.nblocks)
    image = clean[REF_H]
    
    # -- normalize --
    clean -= clean.min()
    clean /= clean.max()
    tv_utils.save_image(clean,"fast_unet_clean.png",normalize=True)
    tv_utils.save_image(image,"fast_unet_image.png",normalize=True)

    # -- apply noise --
    noisy = noise_xform(clean)
    
    # aveT = torch.mean(burst,dim=0)
    # print("Ave MSE: %2.3e" % images_to_psnrs(aveT.unsqueeze(0),image.unsqueeze(0)))
    record = []

    block_search_space = get_block_arangements(cfg.nblocks,cfg.nframes)
    for prop in tqdm(block_search_space):
        clean_prop = clean[prop]
        noisy_prop = noisy[prop]

        model = UNet_n2n(1)
        cfg.init_lr = 1e-4
        optim = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0.9,0.99))

        train(cfg,image,clean_prop,noisy_prop,model,optim)
        results = test(cfg,image,clean_prop,noisy_prop,model)
        record.append(results)
    record = pd.DataFrame(record)
    record_fn = FAST_UNET_DIR / "default.csv"
    print(f"Writing record to {record_fn}")
    record.to_csv(record_fn)
