

from easydict import EasyDict as edict

import align.nnf
import align.combo.optim
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def config():
    cfg = edict()
    cfg = None
    cfg.random_seed = 123
    raise NotImplemented("")
    return cfg
