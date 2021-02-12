

# python imports
from pathlib import Path

# pytorch imports
import torch

# project imports

def load_optimizer(cfg,model):
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=cfg.init_lr)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=cfg.init_lr,
    #                             momentum=0.9,
    #                             weight_decay=1e-4)
    return optimizer


