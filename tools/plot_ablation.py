# python imports
import sys
sys.path.append("./lib")
sys.path.append("./tools")
import numpy as np
import numpy.random as npr
from easydict import EasyDict as edict
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# project imports
import settings
from example_static import test_disent
from aws_denoising import get_denoising_cfg
from pyutils.misc import write_pickle,read_pickle
from pyutils.plot import add_legend

def get_exps():
    exps = edict()

    name = "b83e826d-06ae-4ea1-9c84-ad433a18179a"
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, 1-1 compare, milestone_lr @ [50,400] epochs"
    exps[name].N = 2
    exps[name].batch_size = 256
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = False
    exps[name].hyper_h = 0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}

    # name = "2d030fa1-b717-4315-b977-4edb34fc7988"
    name = ""
    exps[name] = edict()
    exps[name].desc = "simclr loss img rec, 1-1 compare, milestone_lr @ [50,400] epochs"
    exps[name].N = 2
    exps[name].batch_size = 256
    exps[name].img_loss_type = 'simclr'
    exps[name].share_enc = False
    exps[name].hyper_h = 0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}


    name = "314295c7-ee7d-416c-85c7-97626c118e6e"
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, N-N compare (noshare h), milestone_lr @ [50,400] epochs"
    exps[name].N = 15
    exps[name].batch_size = 256
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = False
    exps[name].hyper_h = 0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400]}


    name = "941bef1a-027c-41a1-b851-8c088cac7efc"
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, N-N compare (share h), milestone_lr @ [50,400] epochs"
    exps[name].N = 5
    exps[name].batch_size = 256
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = True
    exps[name].hyper_h = 0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}


    name = "6f62345f-696f-4699-a304-4536deb8c365"
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, simclr loss embeddings, N-N compare (share h), milestone_lr @ [50,400] epochs"
    exps[name].N = 5
    exps[name].batch_size = 256
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = True
    exps[name].hyper_h = 1.0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}


    name = "45d4f791-06cd-4dfc-aedb-9afc26211720"
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, simclr loss embeddings, N-N compare (share h), milestone_lr @ [50,400] epochs"
    exps[name].N = 15
    exps[name].batch_size = 256
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = True
    exps[name].hyper_h = 1.0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}

    name = "" #67b2b8b3-4cc4-41a3-a2d2-e3bb07ec0e6f # what happened to this?
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, simclr loss embeddings, N-N compare (share h), milestone_lr @ [50,400] epochs"
    exps[name].N = 5
    exps[name].batch_size = 256
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = True
    exps[name].hyper_h = 2.0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}

    name = "2244547d-1d85-4344-8354-bf7d6d976630"
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, simclr loss embeddings, N-N compare (share h), milestone_lr @ [120,150,350] epochs"
    exps[name].N = 5
    exps[name].batch_size = 256
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = True
    exps[name].hyper_h = 1.0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[120,150,400],'gamma':0.1}
    # step until 245, then onecycle

    "c42ebeb2-bdce-4356-ad7c-f40209f78637" 
    # step until 245, then onecycle until 270, then step
    name = ""
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, simclr loss embeddings, N-N compare (share h)"
    exps[name].N = 5
    exps[name].batch_size = 256
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = True
    exps[name].hyper_h = 10.0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}

    "188a12f6-4795-4637-821c-f070a529c63e" # share, no loss
    name = ""
    exps[name] = edict()
    exps[name].desc = "simclr loss img rec, N-N compare"
    exps[name].N = 15
    exps[name].batch_size = 56
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = False
    exps[name].hyper_h = 0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}

    name = ""
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, simclr loss img rec, N-N compare"
    exps[name].N = 15
    exps[name].batch_size = 56
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = False
    exps[name].hyper_h = 0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}

    name = ""
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, N-1 compare (share h and aux)"
    exps[name].N = 15
    exps[name].batch_size = 56
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = True
    exps[name].hyper_h = 0
    exps[name].lr_start = 1e-3
    exps[name].lr_policy = "step"
    exps[name].lr_params = {'milestone':[50,400],'gamma':0.1}
    # todo: allow share "aux"

    return exps

def get_exp_cfg(name,gpuid,epoch_num,dataset,exp):
    args = edict()
    args.name = name
    args.gpuid = gpuid
    args.epoch_num = epoch_num
    args.dataset = dataset
    args.noise_level = 0
    args.batch_size = exp.batch_size
    args.img_loss_type = exp.img_loss_type
    args.N = exp.N
    args.share_enc = exp.share_enc
    args.hyper_h = exp.hyper_h
    args.lr_start = exp.lr_start
    args.lr_policy = exp.lr_policy
    args.lr_params = exp.lr_params
    args.new = False
    cfg = get_denoising_cfg(args)
    return cfg
    
def get_cache_filename(name,epoch_num,num=None,use_psnr=False):
    root = Path(f"{settings.ROOT_PATH}/cache/plot_ablation/")
    if not root.exists(): root.mkdir(parents=True)
    if num is None:
        if use_psnr:
            fn = Path(f"{name}_plot_ablation_epoch_{epoch_num}_psnr.pkl")
        else:
            fn = Path(f"{name}_plot_ablation_epoch_{epoch_num}.pkl")
        path = root / fn
    else:
        if use_psnr:
            fn = Path(f"{name}_plot_ablation_epoch_{epoch_num}_psnr_{num}.pkl")
        else:
            fn = Path(f"{name}_plot_ablation_epoch_{epoch_num}_{num}.pkl")
        path = root / fn
    return path

def get_unique_cache_filename(name,epoch_num,use_psnr=False):
    root = Path(f"{settings.ROOT_PATH}/cache/plot_ablation/")
    if use_psnr:
        fn = Path(f"{name}_plot_ablation_epoch_{epoch_num}_psnr.pkl")
    else:
        fn = Path(f"{name}_plot_ablation_epoch_{epoch_num}.pkl")        
    path = root / fn
    idx = 1
    while path.exists():
        if use_psnr:
            fn = Path(f"{name}_plot_ablation_epoch_{epoch_num}_psnr_{idx}.pkl")
        else:
            fn = Path(f"{name}_plot_ablation_epoch_{epoch_num}_{idx}.pkl")
        path = root / fn
        idx += 1
    return path

def save_cache_results(losses,name,epoch_num,use_psnr=False):
    path = get_unique_cache_filename(name,epoch_num,use_psnr)
    write_pickle(losses,path)
    
def load_cached_results(name,epoch_num,num=None,use_psnr=False):
    path = get_cache_filename(name,epoch_num,num,use_psnr)
    if not path.exists():
        if settings.verbose >= 1:
            print(f"No cache for [{path}]")
            print(f"Running experiment for [{name}]")
        return None
    if settings.verbose >= 2:
        print(f"Loading cache from [{path}]")
    return read_pickle(path)

def plot_noise_level(ax,epoch_grid):
    L = len(epoch_grid)
    mean = np.repeat(1.290e-02,L)
    stddev = np.repeat(1.149e-02,L)
    yerr = 1.96 * stddev
    ax.errorbar(epoch_grid,mean,yerr=stddev,alpha=0.5)

def check_valid(cfg,epoch_num):
    root = Path(f"{settings.ROOT_PATH}/")
    exp_dir = root / Path(f"output/disent/mnist/{cfg.exp_name}/enc_c/")
    fn = exp_dir / Path(f"checkpoint_{epoch_num}.tar")
    return fn.exists()

def main():
    exps = get_exps()

    gpuid = 0
    #epoch_grid = [-1,0,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,175,200,225]
    # epoch_grid = [10,15,20,25,30,35,40,45,50,60,70,80,90,100]
    epoch_grid = [5,25,50,75,100,150,200,250,300]
    dataset = "MNIST"
    use_psnr = False

    # get results
    results = {}
    for name,fields in exps.items():
        if len(name) == 0: continue

        results[name] = edict()
        results[name].means = {}
        results[name].stderrs = {}
        results[name].te_losses = {}

        for epoch_num in epoch_grid:
            losses = load_cached_results(name,epoch_num,use_psnr=use_psnr)
            if losses is not None:
                sepoch = str(epoch_num)
                results[name].means[sepoch] = losses.mean
                results[name].stderrs[sepoch] = losses.stderr
                results[name].te_losses[sepoch] = losses.te_losses
                continue

            cfg = get_exp_cfg(name,gpuid,epoch_num,dataset,fields)
            valid = check_valid(cfg,epoch_num)
            if not valid: continue

            losses = test_disent(cfg,use_psnr=use_psnr)
            sepoch = str(epoch_num)
            results[name].means[sepoch] = losses.mean
            results[name].stderrs[sepoch] = losses.stderr
            results[name].te_losses[sepoch] = losses.te_losses

            save_cache_results(losses,name,epoch_num,use_psnr)
        
        # report best loss for each experiment
        print(name,results[name].means.items())
        egrid,means = zip(*results[name].means.items())
        if use_psnr:
            idx = np.argmax(means)
            rstr = "Exp {:s} best psnr of {:2.3e} at epoch {:s}"
        else:
            idx = np.argmin(means)            
            rstr = "Exp {:s} best test loss of {:2.3e} at epoch {:s}"
        fmt = (name,means[idx],egrid[idx])
        print(rstr.format(*fmt))
    
    # plot results
    fig,ax = plt.subplots(figsize=(8,8))
    names = []
    for name,result in results.items():
        egrid,means = zip(*result.means.items())
        egrid,stderrs = zip(*result.stderrs.items())
        grid = [int(e) for e in egrid]
        ax.errorbar(grid,means,yerr=stderrs,fmt='+-',
                    label=name,alpha=1.0)
        names.append(name[0:5])
    # plot_noise_level(ax,epoch_grid)
    # names.append("noise")
    ax.set_title("Ablation Experiments: Testing Losses")
    add_legend(ax,"Experiment",names,framealpha=0.0)
    ax.set_yscale("log",nonpositive="clip")
    path = f"{settings.ROOT_PATH}/reports/plot_ablation.png"
    print(f"Saving plot to {path}")
    plt.savefig(path,transparent=True,dpi=300)
    

if __name__ == "__main__":
    main()
