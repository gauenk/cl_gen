
# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tvT
from torchvision.transforms import functional as tvTF
from torchvision import utils as tv_utils

# -- project imports --
from datasets.transforms import get_noise_transform
from n2sim.sim_search import compute_similar_bursts_analysis
from pyutils.misc import images_to_psnrs

# -- [local] project imports --
from n2sim.debug import print_tensor_stats
from .noise_settings import create_noise_level_grid

def test_sim_search(cfg,clean,model):

    # -- init --
    N,B,C,H,W = clean.shape
    cleanBN = rearrange(clean,'n b c h w -> (b n) c h w')
    clean_pil = [tvT.ToPILImage()(cleanBN[i]+0.5).convert("RGB") for i in range(B*N)]
    ps = cfg.byol_patchsize
    unfold = nn.Unfold(ps,1,0,1)

    # -- start loop --
    psnrs = {}
    noisy_grid = create_noise_level_grid(cfg)
    for noise_params in noisy_grid:
        
        # -- get noisy images --
        cfg.noise_type = noise_params.ntype
        cfg.noise_params.ntype = cfg.noise_type
        cfg.noise_params[cfg.noise_type] = noise_params
        noise_func = get_noise_transform(cfg.noise_params)
        noisyBN = torch.stack([noise_func(clean_pil[i]) for i in range(B*N)],dim=0)
        # tv_utils.save_image(cleanBN,"cleanBN.png",nrow=N,normalize=True)
        # tv_utils.save_image(noisyBN,"noisyBN.png",nrow=N,normalize=True)

        # -- get noisy patches --
        pad = ps//2
        noisy_pad = F.pad(noisyBN,(pad,pad,pad,pad),mode='reflect')
        patches = unfold(noisy_pad)
        patches = rearrange(patches,'bn (c ps1 ps2) r -> (bn r) c ps1 ps2',ps1=ps,ps2=ps)

        # -- get noisy features --
        patches = patches.cuda(non_blocking=True)
        features_p = model(patches,return_embedding=True)
        noisy = rearrange(noisyBN,'(b n) c h w -> n b c h w',b=B)
        features = rearrange(features_p,'(b n h w) f -> n b f h w',b=B,n=N,h=H)

        # -- construct similar image --
        query = edict()
        query.pix = noisy[[0]]
        query.ftr = features[[0]]
        query.shape = query.pix.shape

        database = edict()
        database.pix = noisy[[1]]
        database.ftr = features[[1]]
        database.shape = database.pix.shape

        clean_db = edict()
        clean_db.pix = clean[[1]]
        clean_db.ftr = clean_db.pix
        clean_db.shape = clean_db.pix.shape

        sim_outputs = compute_similar_bursts_analysis(cfg,query,database,clean_db,1,
                                                      patchsize=cfg.sim_patchsize,
                                                      shuffle_k=False,kindex=None,
                                                      only_middle=cfg.sim_only_middle,
                                                      db_level='frame',
                                                      search_method=cfg.sim_method,
                                                      noise_level=None)

        # -- compute psnr --
        ref = clean[0]
        clean_sims = sim_outputs[1][0,:,0]
        psnrs_np = images_to_psnrs(ref.cpu(),clean_sims.cpu())
        psnrs[noise_params.name] = edict()
        psnrs[noise_params.name].psnrs = psnrs_np
        psnrs[noise_params.name].ave = np.mean(psnrs_np)
        psnrs[noise_params.name].std = np.std(psnrs_np)
        psnrs[noise_params.name].min = np.min(psnrs_np)
        psnrs[noise_params.name].max = np.max(psnrs_np)
        # print(noise_params.name,psnrs[noise_params.name])

    return psnrs


def test_sim_search_pix(cfg,clean,model):

    # -- init --
    N,B,C,H,W = clean.shape
    cleanBN = rearrange(clean,'n b c h w -> (b n) c h w')
    clean_pil = [tvT.ToPILImage()(cleanBN[i]+0.5).convert("RGB") for i in range(B*N)]
    ps = cfg.byol_patchsize
    unfold = nn.Unfold(ps,1,0,1)

    # -- start loop --
    psnrs = {}
    noisy_grid = create_noise_level_grid(cfg)
    for noise_params in noisy_grid:
        
        # -- get noisy images --
        cfg.noise_type = noise_params.ntype
        cfg.noise_params.ntype = cfg.noise_type
        cfg.noise_params[cfg.noise_type] = noise_params
        noise_func = get_noise_transform(cfg.noise_params)
        noisyBN = torch.stack([noise_func(clean_pil[i]) for i in range(B*N)],dim=0)
        noisy = rearrange(noisyBN,'(b n) c h w -> n b c h w',b=B)

        # -- construct similar image --
        query = edict()
        query.pix = noisy[[0]]
        query.ftr = noisy[[0]]
        query.shape = query.pix.shape

        database = edict()
        database.pix = noisy[[1]]
        database.ftr = noisy[[1]]
        database.shape = database.pix.shape

        clean_db = edict()
        clean_db.pix = clean[[1]]
        clean_db.ftr = clean_db.pix
        clean_db.shape = clean_db.pix.shape

        sim_outputs = compute_similar_bursts_analysis(cfg,query,database,clean_db,1,
                                                      patchsize=cfg.sim_patchsize,
                                                      shuffle_k=False,kindex=None,
                                                      only_middle=cfg.sim_only_middle,
                                                      db_level='frame',
                                                      search_method=cfg.sim_method,
                                                      noise_level=None)

        # -- compute psnr --
        ref = clean[0]
        clean_sims = sim_outputs[1][0,:,0]
        psnrs_np = images_to_psnrs(ref.cpu(),clean_sims.cpu())
        psnrs[noise_params.name] = edict()
        psnrs[noise_params.name].psnrs = psnrs_np
        psnrs[noise_params.name].ave = np.mean(psnrs_np)
        psnrs[noise_params.name].std = np.std(psnrs_np)
        psnrs[noise_params.name].min = np.min(psnrs_np)
        psnrs[noise_params.name].max = np.max(psnrs_np)
        # print(noise_params.name,psnrs[noise_params.name])

    return psnrs


def print_psnr_results(psnrs,title):
    write_info = (title,psnrs['clean'].ave,psnrs['clean'].std,
                  psnrs['clean'].min,psnrs['clean'].max)
    title = "%s" % title
    sep = "|"

    name = 'clean'
    name_str = "[c]:"
    ave_std = "%2.2f +/- %2.2f" % (psnrs[name].ave,psnrs[name].std)
    min_max = "(%2.2f,%2.2f)" % (psnrs[name].min,psnrs[name].max)
    w_str = f"{title: >15} {name_str: <10} {ave_std: >18} {sep: ^4} {min_max: >12}"
    print(w_str)

    title = ""
    name = 'g-75p0'
    name_str = "[g-75]:"
    ave_std = "%2.2f +/- %2.2f" % (psnrs[name].ave,psnrs[name].std)
    min_max = "(%2.2f,%2.2f)" % (psnrs[name].min,psnrs[name].max)
    w_str = f"{title: >15} {name_str: <10} {ave_std: >18} {sep: ^4} {min_max: >12}"
    print(w_str)

    title =""
    name = 'pn-4p0-0p0'
    name_str = "[pn-4]:"
    ave_std = "%2.2f +/- %2.2f" % (psnrs[name].ave,psnrs[name].std)
    min_max = "(%2.2f,%2.2f)" % (psnrs[name].min,psnrs[name].max)
    w_str = f"{title: >15} {name_str: <10} {ave_std: >18} {sep: ^4} {min_max: >12}"
    print(w_str)
    
    
