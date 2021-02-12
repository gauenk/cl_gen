
# -- python imports --
from pathlib import Path
import numpy as np

# -- pytorch imports --
from torch.optim import lr_scheduler as lr_sched

# -- project imports --


def load_scheduler(cfg,model,optimizer):
    scheduler = lr_sched.ReduceLROnPlateau(optimizer,
                                           mode="min",
                                           patience = 5,
                                           factor=1./np.sqrt(10))
    return scheduler
