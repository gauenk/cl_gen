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
import sys,os,shutil,re,json
sys.path.append("./lib/")
sys.path.append("./tools/")
import argparse,uuid
from easydict import EasyDict as edict
from pathlib import Path
import torch
import numpy as np

# project imports
import settings
from settings import ROOT_PATH
from pyutils.cfg import get_cfg
from example_static import train_disent_exp,test_disent_over_epochs,test_disent_examples_over_epochs,plot_noise_floor,test_disent_examples

def get_args():
    parser = argparse.ArgumentParser(description="Run static denoising on AWS")
    parser.add_argument("--mode", type=str, default="train",
                        help="train or test models")
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
    parser.add_argument("--img-loss-type", type=str, default='l2',
                        help="loss type [l2 or simclr]")
    parser.add_argument("--share-enc", dest="share_enc",
                        action='store_true',help="do we share encodings?")
    parser.add_argument("--hyper_h",  type=float, default=0.,
                        help="hyperparmeter for CL loss on encodings.")
    parser.add_argument("--new",  action='store_true',
                        help="when running from an old experiment, do we create a new experiment file?")
    parser.add_argument("--lr-start",  type=float, default=1e-3,
                        help="initial learning rate")
    parser.add_argument("--lr-policy",  type=str, default="step",
                        help="scheduler policy for optimization")
    parser.add_argument("--lr-params",  type=json.loads,
                        default='{"milestone":[50,500],"gamma":0.1}',
                        help="parameters for scheduler")
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
        
def get_denoising_cfg(args):
    cfg = get_cfg()
    if args.new is True: setup_new_exp(args)

    cfg.cl.device = torch.device("cuda:{}".format(args.gpuid))
    cfg.cls.device = cfg.cl.device


    # set the name
    cfg.exp_name = args.name
    if cfg.exp_name is None:
        cfg.exp_name = str(uuid.uuid4())

    cfg.disent = edict()
    cfg.disent.epochs = 1200
    cfg.disent.device = cfg.cl.device

    cfg.disent.load = args.epoch_num > 0
    cfg.disent.epoch_num = args.epoch_num

    cfg.disent.dataset = edict()
    cfg.disent.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.disent.dataset.n_classes = 10
    cfg.disent.dataset.name = args.dataset
    cfg.disent.noise_level = args.noise_level
    cfg.disent.N = args.N
    cfg.disent.img_loss_type = args.img_loss_type
    cfg.disent.share_enc = args.share_enc
    cfg.disent.hyper_h = args.hyper_h
    cfg.disent.lr = edict()
    cfg.disent.lr.start = args.lr_start
    cfg.disent.lr.policy = args.lr_policy
    cfg.disent.lr.params = args.lr_params

    dsname = cfg.disent.dataset.name.lower()
    model_path = Path(f"{settings.ROOT_PATH}/output/disent/{dsname}/{cfg.exp_name}")
    optim_path = Path(f"{settings.ROOT_PATH}/output/disent/{dsname}/{cfg.exp_name}/optim/")
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
    cfg.disent.log_interval = 50
    cfg.disent.random_crop = True

    if cfg.disent.dataset.name.lower() == "mnist":
        cfg.disent.n_channels = 1
    else:
        cfg.disent.n_channels = 3


    cfg.disent.agg_enc_fxn = 'mean'
    if cfg.disent.share_enc is False:
        cfg.disent.agg_enc_fxn = 'id'

    # info = {'noise':args.noise_level,'N':args.N,
    #         'dataset':args.dataset,'batch_size':args.batch_size}
    # write_settings(cfg.exp_name,info)
    # print(info)
    return cfg


def copy_checkpoint_files(dsname,epoch_num,old_name,new_name):
    base_output = Path(f"{settings.ROOT_PATH}/output/disent/")
    old_path = base_output / Path(f"{dsname}/{old_name}")
    new_path = base_output / Path(f"{dsname}/{new_name}")
    new_path.mkdir()
    rstr = ".*/checkpoint_(?P<num>-?[0-9]+)\.tar"
    for fn in old_path.glob("*/*"):
        sfn = str(fn)
        num = int(re.match(rstr,sfn).groupdict()['num'])
        if num <= epoch_num: # copy the file if meet criteria
            mid_stem = fn.parent.stem
            stem = Path("checkpoint_{}.tar".format(num))
            old_path_fn = old_path / mid_stem / stem
            new_path_dir = new_path / mid_stem
            if not new_path_dir.exists(): new_path_dir.mkdir()
            new_path_fn = new_path_dir / stem
            shutil.copy(old_path_fn,new_path_fn)

    # shutil.copytree(old_path,new_path)

def setup_new_exp(args):
    if args.name is None:
        raise ValueError("To run a new experiment we must use an old one!")
    dsname = args.dataset.lower()
    epoch_num = args.epoch_num
    old_name = args.name
    new_name = str(uuid.uuid4())    
    print(f"Copying experiment results from {old_name} to {new_name}")
    copy_checkpoint_files(dsname,epoch_num,old_name,new_name)
    print("Creating config with new uuid.")
    args.name = new_name

if __name__ == "__main__":

    args = get_args()
    cfg = get_denoising_cfg(args)
    print(f"Running aws_denoising experiments with mode {args.mode}")
    print(f"Experiment named {cfg.exp_name}")


    if args.mode == "train":
        # plot_noise_floor(cfg)
        train_disent_exp(cfg)
    elif args.mode == "test":
        cfg.disent.load = True
        if cfg.disent.epoch_num == 0:
            epoch_num_list = list(range(0,100+1,10)) + [-1]
            print(epoch_num_list)
            test_disent_over_epochs(cfg,epoch_num_list)
            test_disent_examples_over_epochs(cfg,epoch_num_list)
        else:
            test_disent_examples(cfg)
