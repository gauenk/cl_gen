

# python imports
from pathlib import Path

# pytorch imports
import torch

# project imports

def load_optimizer(cfg,model):
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0.1,0.99))
    return optimizer


def load_optimizer_kpn(cfg,model):
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0.9,0.99))
    return optimizer

def load_optimizer_gan(cfg,model):
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0,0.9))
    return optimizer

def load_optim_fp(cfg,optimizer,fp,rank):
    if cfg.use_ddp:
        map_location = {'cuda:%d' % 0, 'cuda:%d' % rank}
    else:
        map_location = 'cuda:%d' % rank
    print(f"Loading model filepath [{fp}]")
    state = torch.load(fp, map_location=map_location)
    optimizer.load_state_dict(state)
    return optimizer


