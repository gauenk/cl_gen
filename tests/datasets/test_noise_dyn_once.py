

# -- python --
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- our faiss func --
import faiss,sys
sys.path.append("/home/gauenk/Documents/faiss/contrib/")
import nnf_utils as nnf_utils

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

# -- save path for viz --
ROOT = Path(f"{settings.ROOT_PATH}")
TEST_PATH = ROOT / "output/tests/datasets/test_noise_dyn_once/"
if not TEST_PATH.exists(): TEST_PATH.mkdir(parents=True)

def get_cfg_defaults():
    cfg = edict()

    # -- frame info --
    cfg.nframes = 3
    cfg.frame_size = 128

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 0
    cfg.dataset.nsamples = 100
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

    # -- search config --
    cfg.nblocks = 5
    cfg.patchsize = 3

    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

def convert_keys(sample):

    translate = {'noisy':'dyn_noisy',
                 'burst':'dyn_clean',
                 'snoisy':'static_noisy',
                 'sburst':'static_clean',
                 'ref_flow':'flow_gt',
                 'seq_flow':'seq_flow',
                 'index':'image_index'}

    if is_converted(sample,translate): return sample
    for field1,field2 in translate.items():
        sample[field2] = sample[field1]
        if field2 != field1: del sample[field1]
    return sample

def test_noise_once():

    # -- run exp --
    cfg = get_cfg_defaults()
    cfg.random_seed = 123
    nbatches = 20
    ref = cfg.nframes//2

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load single dataset --
    print("a \"many\" dataset.")
    cfg.dynamic_info.sim_once = False
    cfg.noise_params.sim_once = False
    many_data,loaders = load_dataset(cfg,"dynamic")
    
    # -- load many dataset --
    print("yes a sim once.")
    cfg.dynamic_info.sim_once = False
    cfg.noise_params.sim_once = True
    once_data,loaders = load_dataset(cfg,"dynamic")

    # -- the actual test --
    nsamples = 10
    nchecks = 4
    for i in range(nsamples):

        # -- collect samples --
        many_bursts = []
        once_bursts = []
        for j in range(nchecks):
            many_sample = many_data.tr[i]
            once_sample = once_data.tr[i]
            many_burst = many_sample['dyn_noisy'][ref]
            once_burst = once_sample['dyn_noisy'][ref]
            many_bursts.append(many_burst)
            once_bursts.append(once_burst)
        many_bursts = torch.stack(many_bursts)
        once_bursts = torch.stack(once_bursts)

        # -- confirm change for "many" --
        for j1 in range(nchecks):
            for j2 in range(nchecks):
                if j1 == j2: continue
                delta = many_bursts[j1] - many_bursts[j2]
                delta = torch.mean(delta**2).item()
                assert delta > 0, "not equal for many!"
        print("Passed \"many\".")
        
        # -- confirm static for "once" --
        save_image(once_bursts,"once_bursts.png")
        for j1 in range(nchecks):
            for j2 in range(nchecks):
                if j1 == j2: continue
                delta = once_bursts[j1] - once_bursts[j2]
                delta = torch.sum(delta**2).item()
                assert delta == 0, "must be equal for many!"
        print("Passed \"once\".")

def test_dyn_once():

    # -- run exp --
    cfg = get_cfg_defaults()
    cfg.random_seed = 123
    nbatches = 20

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load single dataset --
    print("a \"many\" dataset.")
    cfg.dynamic_info.sim_once = False
    many_data,loaders = load_dataset(cfg,"dynamic")

    # -- load many dataset --
    print("yes a sim once.")
    cfg.dynamic_info.sim_once = True
    once_data,loaders = load_dataset(cfg,"dynamic")


        
