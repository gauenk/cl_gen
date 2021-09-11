
"""
A file containings many common
functions used for testing. 

This should be kept as simple as 
possible to minimize deps

"""

import settings
from easydict import EasyDict as edict
import torch
import random
import numpy as np

def get_cfg_defaults():
    cfg = edict()

    # -- frame info --
    cfg.nframes = 3
    cfg.frame_size = 32

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 1
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

def is_converted(sample,translate):
    for key1,key2 in translate.items():
        if not(key2 in sample): return False
    return True

