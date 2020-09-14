
# python imports
import sys,json
sys.path.append("./lib")
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from pathlib import Path

# project imports
import settings
from pyutils.plot import add_legend


def get_exps_a():
    exps = edict()
    base = Path(f"{settings.ROOT_PATH}/reports/noise_5e-1/")

    name = 'n2'
    exps[name] = edict()
    exps[name].exp_name = "d4bae353-f95a-40b4-bfec-0cd78f8b58a3"
    exps[name].path = base / Path(f"N_2_{exps[name].exp_name}/")

    name = 'n5'
    exps[name] = edict()
    exps[name].exp_name = "178c541a-6527-445e-ab61-6991968aa9c9"
    exps[name].path = base / Path(f"N_5_{exps[name].exp_name}/")

    name = 'n10'
    exps[name] = edict()
    exps[name].exp_name = "deb02a01-3562-4366-ac2f-23cab6dc33de"
    exps[name].path = base / Path(f"N_10_{exps[name].exp_name}/")

    name = 'n15'
    exps[name] = edict()
    exps[name].exp_name = "5920effa-aa41-460b-91dc-c5eb86f73394"
    exps[name].path = base / Path(f"N_15_{exps[name].exp_name}/")

    return exps

def get_exps_b():
    exps = edict()
    base = Path(f"{settings.ROOT_PATH}/reports/noise_5e-1/")

    name = "tr_n15_te_n2"
    exps[name] = edict()
    exps[name].exp_name = "5920effa-aa41-460b-91dc-c5eb86f73394"
    exps[name].path = base / Path(f"tr_N_15_on_te_N_2_{exps[name].exp_name}/")

    name = "tr_n2_te_n15"
    exps[name] = edict()
    exps[name].exp_name = "d4bae353-f95a-40b4-bfec-0cd78f8b58a3"
    exps[name].path = base / Path(f"tr_N_2_on_te_N_15_{exps[name].exp_name}/")

    return exps

def get_exp_results(info):
    fn = info.path / Path("results.txt")
    with open(fn,"r") as f:
        results = json.load(f)
    results = {int(t):r for t,r in results.items()}
    # results = {{t:int(k):v for k,v in r.items()} for t,r in results.items()}
    return results

def main():

    exps = get_exps_a()
    nexps = len(exps)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    lnames = []
    for name,info in exps.items():
        print(name)
        results = get_exp_results(info)
        epochs,losses = zip(*results.items())
        ax.plot(epochs,losses,'o-',label=name,alpha=0.5)
        # epochs,means = zip(*results['means'].items())
        # epochs,stderrs = zip(*results['stderrs'].items())
        # ax.errorbar(epochs,means,yerr=stderrs,'o-',label=name,alpha=0.5)
        lnames.append(name)
    add_legend(ax,"model",lnames)
    plt.savefig("exp_plot_test_results_a.png")

    plt.cla()
    plt.clf()

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
