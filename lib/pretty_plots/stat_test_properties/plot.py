# -- python --
import numpy as np
import pandas as pd
from pathlib import Path

# -- python plotting --
import matplotlib
import seaborn as sns
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import cm as plt_cm

# -- local imports --
from .settings import FONTSIZE,MAX_LEN_XLABEL

def plot_single_sim_group(sims,lgrids,title,fname,group,logx=True):
    

    # -- average over groups --
    ggrid = sims[group].unique()
    means,stds = [],[]
    for gvalue in ggrid:
        filtered = sims[sims[group] == gvalue]
        g_mean = filtered['est_mean'].mean()
        g_std = filtered['est_std'].mean()
        means.append(g_mean)
        stds.append(g_std)

    # -- catch a zero if logx --
    eps = 10e-18
    glabels = ggrid.copy()
    if logx: ggrid[ggrid == 0] = eps

    # -- plot --
    fig,ax = plt.subplots(figsize=(8,4))
    ax.errorbar(ggrid,means,yerr=stds)
    if logx: plt.xscale("log")

    # -- format --
    ax.set_xticks(lgrids.ticks[group])
    ax.set_xticklabels(lgrids.tickmarks_str[group],fontsize=FONTSIZE)
    # -- old xticks --
    # ax.set_xticks(ggrid)
    # if len(ggrid) < MAX_LEN_XLABEL:
    #     print(glabels)
    #     glabels = ["%2.2e" % x for x in glabels]
    #     print(glabels)
    #     ax.set_xticklabels(glabels,fontsize=FONTSIZE,rotation=45,ha="right")
    ax.set_xlabel(f"Parameter [{group}]",fontsize=FONTSIZE)
    ax.set_ylabel("Approx. Prob of Alignment",fontsize=FONTSIZE)
    ax.set_title(title,fontsize=18)

    # -- save --
    DIR = Path("./output/pretty_plots")
    if not DIR.exists(): DIR.mkdir()
    fn =  DIR / f"./stat_test_properties_{fname}_group-{group}.png"
    plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")
    plt.xscale("linear")

