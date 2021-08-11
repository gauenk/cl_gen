
# -- python imports --
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from pathlib import Path

# -- project imports --
from pyplots._settings import FONTSIZE,MAX_LEN_XLABEL
from pyplots.legend import add_legend,add_colorbar
from pyplots.misc import add_jitter


def plot_single_sim_group(input_ax,sims,lgrids,title,fname,group,yinfo=None,
                          logx=True,scatter=False,save_dir=None):
    if save_dir is None:
        save_dir = Path("./")
    
    # -- yaxis default--
    if yinfo is None:
        yinfo = edict()
        yinfo.group = "est"
        yinfo.title = "Approx. Prob of Alignment"
    
    # -- average over groups --
    ggrid = sims[group].unique()
    means,stds = [],[]
    for gvalue in ggrid:
        filtered = sims[sims[group] == gvalue]
        g_mean = filtered[f'{yinfo.group}_mean'].mean()
        g_std = filtered[f'{yinfo.group}_std'].mean()
        means.append(g_mean)
        stds.append(g_std)

    # -- catch a zero if logx --
    eps = 10e-18
    glabels = ggrid.copy()
    if lgrids.logs[group] and logx: ggrid[ggrid == 0] = eps

    # -- create main plot --
    if input_ax is None:
        fig,ax = plt.subplots(figsize=(8,4))
    else: ax = input_ax
    ax.errorbar(ggrid,means,yerr=stds)
    if lgrids.logs[group] and logx: ax.set_xscale("log")

    # -- scatter --
    if scatter:
        x = sims[f'{yinfo.group}_mean'].to_numpy()
        jit_std = 0.01
        ax.scatter(sims[group],add_jitter(x,jit_std),marker='x')


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
    ax.set_ylabel(yinfo.title,fontsize=FONTSIZE)
    ax.set_title(title,fontsize=18)
    ax.minorticks_off()


    # -- save --
    if input_ax is None:
        if not save_dir.exists(): save_dir.mkdir()
        fn =  save_dir / f"./{fname}_group-{group}-{yinfo.group}.png"
        plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
        plt.close('all')
        print(f"Wrote plot to [{fn}]")
    plt.xscale("linear")

