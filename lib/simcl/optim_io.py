"""
Loading and saving optim parameters

"""

# python imports
from easydict import EasyDict as edict
import numpy as np
from pathlib import Path

# pytorch imports
import torch
from apex.parallel import DistributedDataParallel as apex_DDP
from torch.nn.parallel import DistributedDataParallel as th_DDP

# project imports
from layers.simcl import ClBlock
from optimizers import LARS

def load_optimizer(cfg,model):
    lr = get_scaled_lr(cfg)
    params = get_model_params(cfg,model)
    optimizer = get_optimizer_type(cfg,params,lr)
    return optimizer

def get_scaled_lr(cfg):
    lr = cfg.init_lr
    bs_scale = cfg.lr_bs_scale
    if bs_scale == "linear":
        lr *= cfg.batch_size * cfg.world_size * cfg.N
    elif bs_scale == "sqrt":
        lr *= np.sqrt(cfg.batch_size * cfg.world_size * cfg.N )
    elif bs_scale == "linear_256": 
        lr *= cfg.batch_size * cfg.world_size * cfg.N / 256.
    elif bs_scale == "sqrt_256":
        lr *= np.sqrt(cfg.batch_size * cfg.world_size * cfg.N / 256. )
    elif bs_scale == "none": 
        lr = lr
    else:
        msg = f"Uknown batch-size scaling of learning rate [{bs_scale}]"
        raise ValueError(msg)
    print("Scaled initial learning rate set at {:2.3e}".format(lr))
    return lr

def get_model_params(cfg,model):
    params = []
    if isinstance(model,edict):
        for name,model in model.items():
            if cfg.freeze_model[name]: continue
            params += list(model.parameters())
    elif isinstance(model,ClBlock):
        return list(model.parameters())
    elif isinstance(model,th_DDP) or isinstance(model,apex_DDP):
        return list(model.parameters())
        for name,freeze_bool in cfg.freeze_model.items():
            if freeze_bool: continue
            if name == "encoder":
                params += list(model.encoder.parameters())
            elif name == "projector":
                params += list(model.projector.parameters())
            else:
                msg = f"Unknown model parameters to freeze [{name}]"
                raise ValueError(msg)
    else:
        raise TypeError(f"Uknown model type: [{model}]")
    return params

def get_optimizer_type(cfg,params,lr):
    ot = cfg.optim_type
    p = cfg.optim_params    
    optimizer = None
    if ot == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr,**p['sgd'])
    elif ot == "adam":
        p = cfg.optim_params
        optimizer = torch.optim.Adam(params, lr=lr, **p['adam'])
    elif ot == "lars":
        p = cfg.optim_params
        p['lars']['use_apex'] = cfg.use_apex
        optimizer = LARS(params, cfg.epochs, lr=lr, **p['lars'])
    elif ot == "sched":
        if cfg.sched_type == "lwca":
            optimizer = torch.optim.SGD(params, lr=lr,**p['sgd'])
        else:
            optimizer = torch.optim.Adam(params, lr=lr, **p['adam'])
    else:
        raise ValueError(f"Uknown optim_type [{ot}]")


    if cfg.load:
        print("Loading optimizer")
        fn = Path("checkpoint_{}.tar".format(cfg.epoch_num))
        optim_fn = Path(cfg.optim_path) / fn
        map_location = lambda storage, loc: storage.cuda(cfg.rank)
        optimizer.load_state_dict(torch.load(optim_fn, map_location=map_location))
    return optimizer

