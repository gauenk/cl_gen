

# python imports
from pathlib import Path

# pytorch imports
import torch

# project imports

def load_optimizer(cfg,model):
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.init_lr,betas=(0,0.9))
    return optimizer

