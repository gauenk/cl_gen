"""
Run the testing loop for denoising experiment
"""

# local proj imports
from .model_io import load_models
from .optim_io import load_optimizer
from .scheduler_io import load_scheduler
from .config import load_cfg,save_cfg,get_cfg,get_args
from .utils import load_hyperparameters,extract_loss_inputs

def run_test(cfg,rank,models,data,loader):
    pass
