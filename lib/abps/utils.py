

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
from datasets.transforms import get_noise_transform
from n2sim.sim_search import compute_similar_bursts_analysis
from pyutils import images_to_psnrs


def print_tensor_stats(name,tensor):
    stats_fmt = (tensor.min().item(),tensor.max().item(),tensor.mean().item())
    stats_str = "%2.2e,%2.2e,%2.2e" % stats_fmt
    print(f"[{name}]",stats_str)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, model_target, model_online):
    """
    ema_updater: Exponential moving average updater. 
    model_target: The model with ema parameters. 
    model_online: The model online uses sgd.
    """
    for params_online, params_target in zip(model_online.parameters(), model_target.parameters()):
        online_weights, target_weights = params_online.data, params_target.data
        params_target.data = ema_updater.update_average(online_weights, target_weights)


def compute_similar_psnr(cfg,noisy_img,ftr_img,clean,q_index,db_index,crop=False):

    # -- construct similar image --
    query = edict()
    query.pix = noisy_img[[q_index]]
    print(query.pix.shape)
    print(noisy_img.shape,noisy_img[[q_index]].shape,noisy_img[q_index].shape)
    query.ftr = ftr_img[[q_index]]
    query.shape = query.pix.shape

    database = edict()
    database.pix = noisy_img[[db_index]]
    database.ftr = ftr_img[[db_index]]
    database.shape = database.pix.shape

    clean_db = edict()
    clean_db.pix = clean[[db_index]]
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
    if crop:
        ref = tvF.crop(ref,10,10,48,48)
        clean_sims = tvF.crop(clean_sims,10,10,48,48)
    psnrs_np = images_to_psnrs(ref.cpu(),clean_sims.cpu())
    return psnrs_np,clean_sims

