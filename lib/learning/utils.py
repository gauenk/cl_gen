"""
Helper functions for learning with pytorch

"""

# python imports
import os
from pathlib import Path

# pytorch imports
import torch

def save_model(cfg, model, optimizer):
    model_path = Path(cfg.model_path)
    if not model_path.exists(): model_path.mkdir(parents=True)
    out = model_path / "checkpoint_{}.tar".format(cfg.current_epoch)

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)


def save_optim(cfg, optimizer):
    optim_path = Path(cfg.optim_path)
    if not optim_path.exists(): optim_path.mkdir(parents=True)
    out = optim_path / "checkpoint_{}.tar".format(cfg.current_epoch)    
    torch.save(optimizer.state_dict(),out)
