# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tvT
from torchvision.transforms import functional as tvF
from torchvision import utils as tv_utils

# -- project imports --
from pyutils import images_to_psnrs
from datasets.transforms import get_noise_transform
from n2sim.sim_search import compute_similar_bursts_analysis

# -- [local] project imports --
from n2sim.debug import print_tensor_stats
from .noise_settings import create_noise_level_grid,get_keys_noise_level_grid
from .utils import print_tensor_stats

def test_local_features(cfg,clean,model):
    ftr_type = cfg.byol_backbone_name
    return test_local_features_type(cfg,clean,model,[ftr_type])


def test_local_features_type(cfg,clean,model,ftr_types):
    
    # -- sample patches --
    n_samples = 2
    indices = np.random.choice(cfg.frame_size**2,n_samples)
    patches = []
    for index in indices:
        index_window = model.patch_helper.index_window(index,ps=3)
        for nh_index in index_window:
            patches_i = model.patch_helper.gather_local_patches(burst+0.5, nh_index)
            patches.append(patches_i)
    patches = torch.cat(patches,dim=1)

    # -- compute locality score for clean pixels --

    with torch.no_grad():
        for noise_params in noisy_grid:
            
            # -- setup noise xform --
            cfg.noise_type = noise_params.ntype
            cfg.noise_params.ntype = cfg.noise_type
            cfg.noise_params[cfg.noise_type] = noise_params
            noise_func = get_noise_transform(cfg.noise_params,noise_only=True)
    
            # -- apply noise --
            noisy_patches = noise_func(patches) # shape = (r n b nh_size^2 c ps_B ps_B)
    
            # -- get features --
            for ftype in ftr_types:
                features = get_features(cfg,noisy_patches,model,ftype)
            
                # -- some debugging code --
                vis = False
                if vis: vis_noisy_features(cfg,noisy_img,ftr_img,clean,ftype)

                testing_indexing = False
                if testing_indexing:
                    test_patch_helper_indexing(cfg,noisy_img,ftr_img,clean,ftype)

                # -- construct similar image --
                psnrs_np = compute_similar_psnr(cfg,noisy_img,ftr_img,clean)
        
                # -- compute psnr --
                psnrs[ftype][noise_params.name] = edict()
                psnrs[ftype][noise_params.name].psnrs = psnrs_np
                compute_psnrs_summary(psnrs[ftype][noise_params.name])
                # psnrs[ftype][noise_params.name].ave = np.mean(psnrs_np)
                # psnrs[ftype][noise_params.name].std = np.std(psnrs_np)
                # psnrs[ftype][noise_params.name].min = np.min(psnrs_np)
                # psnrs[ftype][noise_params.name].max = np.max(psnrs_np)
                del ftr_img
    
    return psnrs


def get_features(cfg,noisy_patches,model,ftr_type):
    if ftr_type == "unet":
        return get_unet_features(cfg,noisy_patches,model)
    elif ftr_type == "attn":
        return get_attn_features(cfg,noisy_patches,model)
    else: raise ValueError(f"Uknown feature type [{ftr_type}]")

def get_pixel_features(cfg,noisy_patches):
    f_mid = cfg.byol_nh_size**2//2
    p_mid = cfg.byol_patchsize//2
    noisy_img = noisy_patches[:,:,:,f_mid,:,p_mid,p_mid]
    noisy_img = rearrange(noisy_img,'(h w) n b c -> n b c h w',h=cfg.frame_size)
    return noisy_img

def get_attn_features(cfg,noisy_patches,model):
    # -- get noisy features a --
    noisy_a = rearrange(noisy_patches,'r n b l c h w -> n (r b) l c h w')
    # noisy_a = rearrange(noisy_a,'n b l c h w -> (n b l) c h w') # "form_input_patches" replaces this
    noisy_inputs = model.patch_helper.form_input_patches(noisy_a)
    embeddings_0 = model(noisy_inputs,return_embedding=True)
    ftr_img_0 = model.patch_helper.embeddings_to_image(embeddings_0)

    # -- get noisy features b --
    noisy_b = torch.flip(noisy_patches,dims=(1,)) # reverse
    # noisy_b = noisy_patches
    noisy_b = rearrange(noisy_b,'r n b l c h w -> n (r b) l c h w')
    # noisy_b = rearrange(noisy_b,'n b l c h w -> (n b l) c h w') # "form_input_patches" replaces this
    noisy_inputs = model.patch_helper.form_input_patches(noisy_b)
    embeddings_1 = model(noisy_b,return_embedding=True)
    ftr_img_1 = model.patch_helper.embeddings_to_image(embeddings_1)

    # -- stack ftr images --
    ftr_img = torch.cat([ftr_img_0,ftr_img_1],dim=0)
    return ftr_img

def get_unet_features(cfg,noisy_patches,model):

    # -- get noisy features a --
    noisy_a = rearrange(noisy_patches,'r n b l c h w -> n (r b) l c h w')
    noisy_inputs = model.patch_helper.form_input_patches(noisy_a)
    print("ni",noisy_inputs.shape)
    embeddings_0 = model(noisy_inputs,return_embedding=True)
    print("e0",embeddings_0.shape)

    # -- get noisy features b --
    noisy_b = torch.flip(noisy_patches,dims=(1,))
    noisy_b = rearrange(noisy_b,'r n b l c h w -> n (r b) l c h w')
    noisy_inputs = model.patch_helper.form_input_patches(noisy_b)
    embeddings_1 = model(noisy_inputs,return_embedding=True)

    # -- stack ftr images --
    features = torch.cat([embeddings_0,embeddings_1],dim=0)
    return features


