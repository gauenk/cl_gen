"""
Training the denoising project on AWS

identify each model version by:
- the number of transformations
- the noise level

todo:
- train each model for 1000 epochs
- download losses across epochs on each dataset split
- download the weights for 1000 epochs
- download exp_name settings

"""

# python imports
import sys,os
sys.path.append("./lib/")
sys.path.append("./tools/")
import argparse,uuid
from easydict import EasyDict as edict
from pathlib import Path
import torch

# project imports
import settings
from settings import ROOT_PATH
from pyutils.cfg import get_cfg
from example_static import train_disent_exp

def get_args():
    parser = argparse.ArgumentParser(description="Run static denoising on AWS")
    parser.add_argument("--noise-level", type=float, default=1e-2,
                        help="noise level for each input image")
    parser.add_argument("--N", type=int, default=2,
                        help="number of noisy images to generate")
    parser.add_argument("--dataset", type=str, default="MNIST",
                        help="which dataset to create")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="how big are the batch sizes")
    parser.add_argument("--gpuid", type=int, default=0,
                        help="which gpu?")
    parser.add_argument("--epoch-num", type=int, default=0,
                        help="resume training from epoch-num")
    parser.add_argument("--name", type=str, default=None,
                        help="experiment name")
    args = parser.parse_args()
    return args
    
def write_settings(exp_name,settings):
    fn = Path(f"{ROOT_PATH}/uuid_lookup.txt")
    wstr = f"exp_name: {exp_name}\n"
    for key,val in settings.items():
        if isinstance(val,float):
            val = "{:.3e}".format(val)
            wstr += f"{key}: {val}\n"
        elif isinstance(val,int):
            val = "{:d}".format(val)
            wstr += f"{key}: {val}\n"
        elif isinstance(val,str):
            wstr += f"{key}: {val}\n"
        else:
            raise TypeError("Uknown settings type {}".format(type(val)))
    with open(fn,'a+') as f:
        f.write(wstr)
        
def main():
    args = get_args()
    cfg = get_cfg()

    cfg.cl.device = torch.device("cuda:{}".format(args.gpuid))
    cfg.cls.device = cfg.cl.device


    # set the name
    cfg.exp_name = args.name
    if cfg.exp_name is None:
        cfg.exp_name = str(uuid.uuid4())

    cfg.disent = edict()
    cfg.disent.epochs = 500

    cfg.disent.load = args.epoch_num > 0
    cfg.disent.epoch_num = args.epoch_num

    cfg.disent.dataset = edict()
    cfg.disent.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.disent.dataset.n_classes = 10
    cfg.disent.dataset.name = args.dataset
    cfg.disent.noise_level = args.noise_level
    cfg.disent.N = args.N


    model_path = Path(f"{settings.ROOT_PATH}/output/disent/{cfg.exp_name}/")
    optim_path = Path(f"{settings.ROOT_PATH}/output/disent/{cfg.exp_name}/optim/")
    if not model_path.exists(): model_path.mkdir(parents=True)
    cfg.disent.model_path = model_path
    cfg.disent.optim_path = optim_path
    
    cfg.disent.workers = 1
    cfg.disent.batch_size = args.batch_size
    cfg.disent.global_step = 0
    cfg.disent.device = cfg.cl.device
    cfg.disent.current_epoch = 0
    cfg.disent.checkpoint_interval = 1
    cfg.disent.test_interval = 5
    cfg.disent.log_interval = 1

    info = {'noise':args.noise_level,'N':args.N,
            'dataset':args.dataset,'batch_size':args.batch_size}
    print(info)
    write_settings(cfg.exp_name,info)
    train_disent_exp(cfg)


if __name__ == "__main__":
    main()
