
# -- python imports --
from pathlib import Path
from easydict import EasyDict as edict

# -- project imports --
from settings import ROOT_PATH
from datasets.common import get_loader

# -- local imports --
from .bsdBurst import BSDBurst


# ----------------------------------
#
#      API Function Call
#
# ----------------------------------

def get_bsdBurst_dataset(cfg,mode):

    # -- select root path --
    root = cfg.dataset.root
    if mode in ["real_motion","dynamic"]:
        root = Path(root)/Path("./bsdBurst/data/")
    elif mode == "synth_motion":
        root = Path(root)/Path("./bsdBurst/BSDBursts/")
    else:
        raise ValueError(f"Unknown BSDBurst mode {mode}")

    # -- create dataset obj --
    data = edict()
    data.tr = BSDBurst(root,cfg.noise_params,cfg.frame_size,cfg.nframes)
    data.val,data.te = data.tr,data.tr

    # -- get and return loader --
    loader = get_loader(cfg,data,cfg.batch_size,mode)
    return data,loader

