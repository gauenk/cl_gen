
# python imports
import sys,json,glob,re
sys.path.append("./lib")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from pathlib import Path

# tensorflow imports
from tensorboard.backend.event_processing import event_accumulator

# project imports
import settings
from pyutils.plot import add_legend
from denoising.exp_utils import load_exp_cache
# from .example_static import get_report_dir

def get_exp_results(info):
    fn = info.path / Path("results.txt")
    with open(fn,"r") as f:
        results = json.load(f)
    results = {int(t):r for t,r in results.items()}
    # results = {{t:int(k):v for k,v in r.items()} for t,r in results.items()}
    return results

def get_experiment_scalars(fn):
    sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 500}
    ea = event_accumulator.EventAccumulator(fn,size_guidance=sg)
    ea.Reload()
    train_scalars = ea.Scalars("Loss/train_epoch")
    val_scalars = ea.Scalars("Loss/val")
    test_scalars = ea.Scalars("Loss/test")
    lr_scalars = ea.Scalars("Misc/learning_rate")
    scalars = {'tr':train_scalars,
               'val':val_scalars,
               'te':test_scalars,
               'lr':lr_scalars,}
    return scalars

def get_run_full_path(run_dir):
    glob_dir = f"{settings.ROOT_PATH}/runs/{run_dir}/*"
    return glob.glob(glob_dir)[0]

def load_experiment_logfiles(version="v1"):
    cfgs = load_exp_cache(version)
    fn = f"cache/run_denoising/runs_{version}.txt"
    with open(fn,"r") as f:
        lines = f.readlines()

    rstr = r"(?P<field>[a-z]+): (?P<val>[a-zA-Z0-9-_]+)"
    experiments = {}
    # hack to parse file
    name = "none"
    idx = 0
    for line in lines:
        idx += 1
        if idx > 176: break
        if len(line) <= 1: # newline brake
            continue
        m = re.match(rstr,line)
        if isinstance(m,type(None)): continue
        m = m.groupdict()
        if m['field'] == "name":
            name = m['val']
            experiments[name] = {}
        elif m['field'] == "run":
            full_path = get_run_full_path(m['val'])
            experiments[name][m['field']] = full_path
            results = get_experiment_scalars(full_path)            
            experiments[name]["results"] = results
        elif m['field'] == "id":
            experiments[name][m['field']] = m['val']
            cfg_id = int(m['val'])
            cfgs[cfg_id]
            experiments[name]['cfg'] = cfgs[cfg_id]

    # remove results with no "run"
    names,infos = zip(*experiments.items())
    for name,info in zip(names,infos):
        if 'results' not in info.keys():
            print(f"removing {name}")
            del experiments[name]

    return experiments

def scalar_max(scalar_list):
    vals = [s.value for s in scalar_list]
    return np.max(vals)

def cache_pd_fn(version):
    fn = f"{settings.ROOT_PATH}/cache/run_denoising/results_{version}.csv"
    return Path(fn)

def check_pd_cache(version):
    fn = cache_pd_fn(version)
    return fn.exists()

def load_pd_cache(version):
    fn = cache_pd_fn(version)
    if fn.exists():
        return pd.read_csv(fn)
    else:
        return None

def save_pd_cache(version,df):
    fn = cache_pd_fn(version)
    df.to_csv(fn)

def get_exp_grid_v1(version="v1"):

    # load from cache
    if check_pd_cache(version):
        return load_pd_cache(version)

    exp_info = load_experiment_logfiles(version="v1")

    data = edict()
    data.N = []
    data.sigma = []
    data.agg_enc_fxn = []
    data.hyper_h = []
    data.te_acc = []

    for name,info in exp_info.items():
        N = info['cfg']['N']
        sigma = info['cfg']['noise_params']['g']['stddev']
        agg_enc_fxn = info['cfg']['agg_enc_fxn']
        hyper_h = info['cfg']['hyper_params']['h']
        te_acc = scalar_max(info['results']['te'])
        
        data.N.append(N)
        data.sigma.append(sigma)
        data.agg_enc_fxn.append(agg_enc_fxn)
        data.hyper_h.append(hyper_h)
        data.te_acc.append(te_acc)

    df = pd.DataFrame(data)

    # save to cache
    save_pd_cache(version,df)

    print(df)
    return df

def plot_x_sigma(df):
    gb_groups = df.groupby('agg_enc_fxn')
    fig,ax = plt.subplots(2,2,figsize=(10,10))
    ax_titles = []
    plt_fmt = {2:'-+r',4:'-+g',8:'-+b',12:'-+k'}
    for i,(agg_enc_fxn,group_i) in enumerate(gb_groups):
        # different plot
        for j,(hyper_h,group_j) in enumerate(group_i.groupby('hyper_h')):
            # different plot
            labels = []
            for k,(N,group_k) in enumerate(group_j.groupby('N')):
                # same plot
                group_k = group_k.sort_values('sigma')
                data = group_k[['sigma','te_acc']].to_numpy()
                ax[i,j].plot(data[:,0],data[:,1],plt_fmt[N])
                labels.append(str(N))
            ax[i,j].set_title(f"agg_enc_fxn: {agg_enc_fxn} | CL Loss Coefficient: {hyper_h}")
            ax[i,j].set_ylabel("PSNR")
            ax[i,j].set_xlabel("sigma")
            ax[i,j] = add_legend(ax[i,j],"N",labels,fontsize=12)
    plt.savefig("./tmp.png")

def plot_x_N(df):
    gb_groups = df.groupby('agg_enc_fxn')
    fig,ax = plt.subplots(2,2,figsize=(10,10))
    ax_titles = []
    plt_fmt = {10:'-+r',50:'-+g',100:'-+b'}
    for i,(agg_enc_fxn,group_i) in enumerate(gb_groups):
        # different plot
        for j,(hyper_h,group_j) in enumerate(group_i.groupby('hyper_h')):
            # different plot
            labels = []
            for k,(sigma,group_k) in enumerate(group_j.groupby('sigma')):
                # same plot
                group_k = group_k.sort_values('N')
                data = group_k[['N','te_acc']].to_numpy()
                ax[i,j].plot(data[:,0],data[:,1],plt_fmt[sigma])
                labels.append(str(sigma))
            ax[i,j].set_title(f"agg_enc_fxn: {agg_enc_fxn} | CL Loss Coefficient: {hyper_h}")
            ax[i,j].set_ylabel("PSNR")
            ax[i,j].set_xlabel("N")
            ax[i,j].set_xticks([2,4,8,12])
            ax[i,j] = add_legend(ax[i,j],"sigma",labels,fontsize=12)
    plt.savefig("./tmp_N.png")

def main():

    df = get_exp_grid_v1(version="v1")
    print(df)
    # plot_x_sigma(df)
    plot_x_N(df)

    # fig,ax = plt.subplots(1,1,figsize=(8,8))
    # lnames = []
    # for name in names:
    #     results = get_exp_results(cfg,name)
    #     epochs,losses = zip(*results.items())
    #     ax.plot(epochs,losses,'o-',label=name,alpha=0.5)
    #     # epochs,means = zip(*results['means'].items())
    #     # epochs,stderrs = zip(*results['stderrs'].items())
    #     # ax.errorbar(epochs,means,yerr=stderrs,'o-',label=name,alpha=0.5)
    #     lnames.append(name)
    # add_legend(ax,"model",lnames)
    # plt.savefig("exp_plot_test_results_a.png")

    # plt.cla()
    # plt.clf()

    # exps = get_exps_b()
    # fig,ax = plt.subplots(1,1,figsize=(8,8))
    # lnames = []
    # for name,info in exps.items():
    #     print(name)
    #     results = get_exp_results(info)
    #     epochs,means = zip(*results['means'].items())
    #     epochs,stderrs = zip(*results['stderrs'].items())
    #     ax.errorbar(epochs,means,yerr=stderrs,'o-',label=name,alpha=0.5)
    #     lnames.append(name)
    # add_legend(ax,"model",lnames)
    # plt.savefig("exp_plot_test_results_b.png")



if __name__ == "__main__":
    print("HI")
    main()
