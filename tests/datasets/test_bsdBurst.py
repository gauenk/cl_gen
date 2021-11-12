

# -- python --
import cv2,tqdm
import random
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch --
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

# -- project --
import settings
from pyutils import save_image
from align import compute_aligned_psnr
from align.nnf import compute_burst_nnf
from patch_search import get_score_function
from align.combo.optim import AlignOptimizer
from align.combo import EvalBlockScores,EvalBootBlockScores
from align.xforms import pix_to_flow,align_from_pix,flow_to_pix,align_from_flow
from datasets import load_dataset
from pyutils.testing import get_cfg_defaults,set_seed,convert_keys,is_converted

def get_cfg_defaults():
    cfg = edict()

    # -- frame info --
    cfg.nframes = 3
    cfg.frame_size = [32,32]
    cfg.frame_size = None

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 0
    cfg.set_worker_seed = True
    cfg.batch_size = 1
    cfg.drop_last = {'tr':True,'val':True,'te':True}
    cfg.noise_params = edict({'pn':{'alpha':10.,'std':0},
                              'g':{'std':25.0},'ntype':'g'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'global'
    cfg.dynamic_info.frame_size = cfg.frame_size
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 2
    cfg.dynamic_info.textured = True
    cfg.random_seed = 0

    # -- combo config --
    cfg.nblocks = 5
    cfg.patchsize = 3
    cfg.score_fxn_name = "bootstrapping"

    return cfg


def create_nnf_for_frame_size_grid(cfg):
    # -- ensure nnf created --
    frame_size_grid = [[64,64],[128,128],[256,256],[512,512],None]
    for frame_size in frame_size_grid:

        # -- load dataset --
        print("load image dataset.")
        cfg.batch_size = 1
        cfg.dataset.name = "bsd_burst"
        cfg.frame_size = frame_size
        data,loaders = load_dataset(cfg,"dynamic")
        print("num of bursts: ",len(loaders.tr))
        nbursts = len(data.tr)
        for burst_index in tqdm.tqdm(range(nbursts)):
            data.tr[burst_index]
    

def test_bsdBurst_dataset():
    
    # -- run exp --
    cfg = get_cfg_defaults()
    cfg.random_seed = 123
    cfg.batch_size = 1
    cfg.frame_size = None
    cfg.dataset.name = "bsd_burst"

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- establish nnf frames are created --
    create_nnf_for_frame_size_grid(cfg)

    # -- save path for viz --
    save_dir = Path(f"{settings.ROOT_PATH}/output/tests/datasets/test_bsdBurst/")
    if not save_dir.exists(): save_dir.mkdir(parents=True)

    # -- load dataset --
    print("load image dataset.")
    data,loaders = load_dataset(cfg,"dynamic")
    print("num of bursts: ",len(loaders.tr))
    nbursts = len(data.tr)

    # -- ensure nnf created --
    for burst_index in tqdm.tqdm(range(nbursts)):
            data.tr[burst_index]

    # -- for image bursts --
    image_iter = iter(loaders.tr)
    for burst_index in range(nbursts):

        # -- sample image --
        sample = next(image_iter)
        noisy = sample['dyn_noisy']
        clean = sample['dyn_clean']
        snoisy = sample['static_noisy']
        sclean = sample['static_clean']
        flow = sample['flow']
        index = sample['image_index'][0][0].item()
        nframes,nimages,c,h,w = noisy.shape
        mid_pix = h*w//2+2*cfg.nblocks
        print(f"Image Index {index}")

        print(noisy.shape)
        print(clean.shape)

        # -- io info --
        image_dir = save_dir / f"index{index}/"
        if not image_dir.exists(): image_dir.mkdir()

        fn = str(save_dir / "./bsdBurst_example.png")
        save_image(sample['dyn_noisy'],fn,normalize=True,vrange=(0.,1.))


