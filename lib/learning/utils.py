"""
Helper functions for learning with pytorch

"""

# python imports
import os
from pathlib import Path

# pytorch imports
import torch
from torch.nn import DataParallel as DP
from apex.parallel import DistributedDataParallel as apex_DDP
from torch.nn.parallel import DistributedDataParallel as th_DDP


def save_model(cfg, model, optimizer=None):
    model_path = Path(cfg.model_path)
    if not model_path.exists(): model_path.mkdir(parents=True)
    out = model_path / "checkpoint_{}.tar".format(cfg.current_epoch)
    print(f"saving model to [{out}]")

    if isinstance(model, DP) or isinstance(model, th_DDP) or isinstance(model, apex_DDP):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)

    if not optimizer is None:
        save_optim(cfg, optimizer)


def save_optim(cfg, optimizer):
    optim_path = Path(cfg.optim_path)
    if not optim_path.exists(): optim_path.mkdir(parents=True)
    out = optim_path / "checkpoint_{}.tar".format(cfg.current_epoch)    
    print(f"saving optim to [{out}]")
    torch.save(optimizer.state_dict(),out)
