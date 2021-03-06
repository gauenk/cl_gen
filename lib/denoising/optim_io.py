"""
Loading and saving optim parameters

"""

# python imports
from easydict import EasyDict as edict
import numpy as np

# pytorch imports
import torch
from apex.parallel import DistributedDataParallel as apex_DDP
from torch.nn.parallel import DistributedDataParallel as th_DDP

# project imports
from layers.denoising import DenoisingBlock
from optimizers import LARS

def get_scaled_lr(cfg):
    lr = cfg.init_lr
    bs_scale = cfg.lr_bs_scale
    if bs_scale == "linear":
        lr *= cfg.batch_size * cfg.world_size * cfg.N / 256. 
    elif bs_scale == "sqrt":
        lr *= np.sqrt(cfg.batch_size * cfg.world_size * cfg.N / 256. )
    elif bs_scale == "none": 
        lr = lr
    else:
        msg = f"Uknown batch-size scaling of learning rate [{bs_scale}]"
        raise ValueError(msg)
    print("Scaled initial learning rate set at {:2.3e}".format(lr))
    return lr

def get_optimizer_type(cfg,names,params,lr):
    ot = cfg.optim_type
    p = cfg.optim_params    
    if ot == "sgd":
        return torch.optim.SGD(params, lr=lr,**p['sgd'])
    elif ot == "adam":
        p = cfg.optim_params
        return torch.optim.Adam(params, lr=lr, **p['adam'])
    elif ot == "lars":
        p = cfg.optim_params
        p['lars']['use_apex'] = cfg.use_apex
        del p['lars']['eta']
        p['lars']['exclude_from_layer_adaptation'] = ['bias','\.bn[0-9]+\.']
        # p['lars']['use_apex'] = cfg.use_apex
        return LARS(names, params, lr=lr, **p['lars'])
    elif ot == "sched":
        if cfg.sched_type == "lwca":
            return torch.optim.SGD(params, lr=lr,**p['sgd'])
        else:
            return torch.optim.Adam(params, lr=lr, **p['adam'])
    else:
        raise ValueError(f"Uknown optim_type [{ot}]")
    
def load_optimizer(cfg,models):
    lr = get_scaled_lr(cfg)
    names,params = get_model_params(cfg,models)
    optimizer = get_optimizer_type(cfg,names,params,lr)
    return optimizer


def get_model_params(cfg,models):
    names,params = [],[]
    
    names,params = zip(*models.named_parameters())
    return names,params
        
    if isinstance(models,edict):

        for name,model in models.items():
            if cfg.freeze_models[name]: continue
            _names,_params = zip(*model.named_parameters())
            names += _names
            params += _params

    elif isinstance(models,th_DDP) or isinstance(models,apex_DDP):

        for name,freeze_bool in cfg.freeze_models.items():
            if freeze_bool: continue
            if name == "encoder":
                _names,_params = zip(*models.module.encoder.named_parameters())
            elif name == "decoder":
                _names,_params = zip(*models.module.decoder.named_parameters())
            elif name == "projector":
                _names,_params = zip(*models.module.projector.named_parameters())
            else:
                msg = f"Unknown model parameters to freeze [{name}]"
                raise ValueError(msg)
            names += _names
            params += _params

    elif isinstance(models,DenoisingBlock):
        for name,freeze_bool in cfg.freeze_models.items():
            if freeze_bool: continue
            if name == "encoder":
                params += list(models.encoder.parameters())
            elif name == "decoder":
                params += list(models.decoder.parameters())
            elif name == "projector":
                params += list(models.projector.parameters())
            else:
                msg = f"Unknown model parameters to freeze [{name}]"
                raise ValueError(msg)
    else:
        raise TypeError(f"Uknown model type: [{models}]")
    return names,params

