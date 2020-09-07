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
    exps[name].desc = "l2 loss img rec, 1-1 compare"
    exps[name].N = 2
    exps[name].batch_size = 128
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = False

    name = "2d030fa1-b717-4315-b977-4edb34fc7988"
    exps[name] = edict()
    exps[name].desc = "simclr loss img rec, 1-1 compare"
    exps[name].N = 2
    exps[name].batch_size = 128
    exps[name].img_loss_type = 'simclr'
    exps[name].share_enc = False

    name = "314295c7-ee7d-416c-85c7-97626c118e6e"
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, N-N compare"
    exps[name].N = 15
    exps[name].batch_size = 56
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = False

    name = ""
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, N-N compare (share h)"
    exps[name].N = 15
    exps[name].batch_size = 56
    exps[name].img_loss_type = 'l2'
    exps[name].share_enc = True

    name = ""
    exps[name] = edict()
    exps[name].desc = "simclr loss img rec, N-N compare"
    exps[name].N = 15
    exps[name].batch_size = 56
    exps[name].img_loss_type = 'l2'

    name = ""
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, simclr loss img rec, N-N compare"
    exps[name].N = 15
    exps[name].batch_size = 56
    exps[name].img_loss_type = 'l2'

    name = ""
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, N-1 compare (share h and aux)"
    exps[name].N = 15
    exps[name].batch_size = 56
    exps[name].img_loss_type = 'l2'
    
    name = ""
    exps[name] = edict()
    exps[name].desc = "l2 loss img rec, N-N compare, simclr loss embeddings"
    exps[name].N = 15
    exps[name].batch_size = 56
    exps[name].img_loss_type = 'l2'

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
    cfg = get_denoising_cfg(args)
    return cfg
    
def test_disent_over_cfg_set(cfg_list):
    means = []
    stderrs = []
    for cfg in cfg_list:
        mean,stderr = test_disent(cfg)
        means.append(mean)
        stderrs.append(stderr)
    return mean,stderr

def get_cache_filename(name,num=None):
    root = Path(f"{settings.ROOT_PATH}/cache/plot_ablation/")
    if not root.exists(): root.mkdir(parents=True)
    if num is None:
        fn = Path(f"{name}_plot_ablation.pkl")
        path = root / fn
    else:
        fn = Path(f"{name}_plot_ablation_{num}.pkl")
        path = root / fn
    return path

def get_unique_cache_filename(name):
    root = Path(f"{settings.ROOT_PATH}/cache/plot_ablation/")
    fn = Path(f"{name}_plot_ablation.pkl")
    path = root / fn
    idx = 1
    while path.exists():
        fn = Path(f"{name}_plot_ablation_{idx}.pkl")
        path = root / fn
        idx += 1
    return path

def save_cache_results(results,name):
    path = get_unique_cache_filename(name)
    write_pickle(results,path)
    
def load_cached_results(name,num=None):
    path = get_cache_filename(name,num)
    if not path.exists():
        print(f"No cache for [{path}]")
        print(f"Running experiment for [{name}]")
        return None
    print(f"Loading cache from [{path}]")
    return read_pickle(path)

def plot_noise_level(ax,epoch_grid):
    L = len(epoch_grid)
    mean = np.repeat(1.290e-02,L)
    stddev = np.repeat(1.149e-02,L)
    yerr = 1.96 * stddev
    ax.errorbar(epoch_grid,mean,yerr=stddev)

def main():
    exps = get_exps()

    gpuid = 0
    epoch_grid = [0,5,10,15,20,25]
    dataset = "MNIST"

    # get results
    results = {}
    for name,fields in exps.items():
        if len(name) == 0: continue
        results[name] = load_cached_results(name)
        if results[name] is not None: continue
            
        results[name] = edict()
        results[name].means = []
        results[name].stderrs = []
        for epoch_num in epoch_grid:
            cfg = get_exp_cfg(name,gpuid,epoch_num,dataset,fields)
            mean,stderr = test_disent(cfg)
            results[name].means.append(mean)
            results[name].stderrs.append(stderr)
        save_cache_results(results[name],name)
    
    # plot results
    fig,ax = plt.subplots(figsize=(8,8))
    names = []
    for name,result in results.items():
        ax.errorbar(epoch_grid,result.means,
                    yerr=result.stderrs,fmt='x-',label=name,alpha=0.5)
        names.append(name[0:5])
    plot_noise_level(ax,epoch_grid)
    names.append("noise")
    ax.set_title("Ablation Experiments: Testing Losses")
    add_legend(ax,"Experiment",names,framealpha=1.0)
    ax.set_yscale("log",nonpositive="clip")
    path = f"{settings.ROOT_PATH}/reports/plot_ablation.png"
    print(f"Saving plot to {path}")
    plt.savefig(path,transparent=False)
    

if __name__ == "__main__":
    main()
