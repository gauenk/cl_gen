
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
from .example_static import get_report_dir

def get_exp_results(info):
    fn = info.path / Path("results.txt")
    with open(fn,"r") as f:
        results = json.load(f)
    results = {int(t):r for t,r in results.items()}
    # results = {{t:int(k):v for k,v in r.items()} for t,r in results.items()}
    return results

def main():

    names = []
    names += ["876f31f8-47d4-468a-8443-ed956304137a"]
    names += [""]
    names += [""]


    fig,ax = plt.subplots(1,1,figsize=(8,8))
    lnames = []
    for name in names:
        results = get_exp_results(cfg,name)
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
