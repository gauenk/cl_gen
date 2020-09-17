"""
Run a single denoising experiment
"""

# python imports
import os,sys
sys.path.append("./lib")
import argparse,json,uuid,yaml,io
from easydict import EasyDict as edict
from pathlib import Path

# project imports
import settings
from denoising import run_localized
from denoising.exp_utils import load_exp_cache

def get_args():
    parser = argparse.ArgumentParser(description="Run a Denoising Experiment")
    parser.add_argument("--cache", type=str, default=None,
                        help="Specify which cache to load from")
    parser.add_argument("--id", type=int, default=None,
                        help="Specify which experiment from the cache to run.")
    parser.add_argument("--gpuid", type=int, default=None,
                        help="Specify the gpu to use.") 
    parser.add_argument("--mode", type=str, default="train",
                        help="Do we train or test?")
    parser.add_argument("--epoch_num", type=int, default=100,
                        help="What epoch do we load if we load?")
    parser.add_argument("--init-lr", type=float, default=None,
                        help="Overwrite the init-lr?")
    msg = """what type of noise do we run experiments with?
        'g': Gaussian Noise
        'll': Low-light Noise
        'msg': Mutli-Scale Gaussian Noise
    """
    parser.add_argument("--noise-type", type=str, default=None, help=msg)
    msg = "How big is each batch?"
    parser.add_argument("--batch-size", type=int, default=None, help=msg)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    cfg = load_exp_cache(args.cache)[args.id]
    cfg.device = 'cuda:%d' % args.gpuid
    cfg.mode = args.mode
    cfg.epoch_num = args.epoch_num
    if cfg.mode == "test":
        cfg.load = True
    if args.init_lr:
        cfg.init_lr = args.init_lr
    if args.noise_type:
        cfg.noise_type = args.noise_type
    if args.batch_size:
        cfg.batch_size = args.batch_size
        cfg.log_interval = 50
    cfg.use_apex = False
    run_localized(cfg=cfg,gpuid=args.gpuid)
