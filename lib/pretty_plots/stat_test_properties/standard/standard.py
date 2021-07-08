# -- python imports --
from numba import jit
import numpy as np
from easydict import EasyDict as edict
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm as plt_cm

# -- project imports --
from pyutils import add_legend,create_named_meshgrid,np_log,numba_unique,write_pickle,read_pickle

# -- local imports --
from .standard_sim import get_standard_sims
from .standard_grid import create_standard_parameter_grid

from pretty_plots.stat_test_properties.misc import skip_with_endpoints
from pretty_plots.stat_test_properties.plot import plot_single_sim_group,stratify_contour_plots,plot_sim_test_pairs,get_agg_mlevels_from_pairs,save_pair_contours

FONTSIZE = 15


def run_standard():
    pgrid,lgrid = create_standard_parameter_grid()
    plot_standard_test(pgrid,lgrid)

def plot_standard_test(pgrid,lgrid):
    sims = get_standard_sims(pgrid,parallel=True)
    title = "Standard Function"
    fname = "standard"
    plot_sim_test(sims,lgrid,title,fname)

def plot_sim_test_pairs_old(ax,sims,lgrids,mlevels,title,fname,field1,field2,
                        add_legend_bool=True):
    
    MAX_LEN_TICKS = 30
    shrink_perc = 0.8

    # -- create field grid --
    f1_unique = list(sims[field1].unique())
    f2_unique = list(sims[field2].unique())
    uniques = [f1_unique,f2_unique]
    names = ["f1","f2"]
    fgrid = create_named_meshgrid(uniques,names)
    
    # -- aggregate over field --
    means = np.zeros((len(f2_unique),len(f1_unique)))
    stds = np.zeros((len(f2_unique),len(f1_unique)))
    aggs = []
    for point in fgrid:
        filter1 = sims[sims[field1] == point.f1]
        filter2 = filter1[filter1[field2] == point.f2]

        # -- create dict --
        agg = edict()
        agg.est_mean = filter2['est_mean'].mean()
        agg.est_std = filter2['est_std'].std()
        agg.f1 = point.f1
        agg.f2 = point.f2

        # -- format samples for contour plot --
        f1_index = f1_unique.index(point.f1)
        f2_index = f2_unique.index(point.f2)
        means[f2_index,f1_index] = agg.est_mean
        stds[f2_index,f1_index] = agg.est_std

        aggs.append(agg)
    
    # -- number of contour levels --
    if mlevels is None:
        vmin = means.min()
        quants = np.linspace(vmin,1.0,30)
        mlevels = np.unique(np.quantile(means,quants))
        # slevels = np.unique(np.quantile(stds,quants))
    
    # -- log axis --
    fields = [field1,field2]
    xgrid,ygrid = f1_unique,f2_unique
    grids = [xgrid,ygrid]
    for idx,field in enumerate(fields):
        if field in ["D","mu2","std"]:
            grids[idx] = np_log(grids[idx])/np_log([10])

    # -- figure size depends on legend inclusion --
    figsize = [6,4]
    if add_legend_bool and (ax is None): figsize[0] = figsize[0]/shrink_perc

    # -- plot contours --
    title = f"Approx. Probability of Correct Alignment [Standard]"
    if ax is None:
        fig,axes = plt.subplots(figsize=figsize)
        axes = [axes]
        axes[0].set_title(title,fontsize=FONTSIZE)
    else:
        axes = [ax]
    cs = axes[0].contourf(grids[0],grids[1],means,levels=mlevels,cmap="plasma")
    

    def translate_axis_labels(labels):
        def translate_label(name):
            if name == "D": return "Patchsize, $D$ [Log-scale]"
            elif name == "std": return "Noise Level, $\sigma^2$ [Log-scale]"
            elif name == "mu2": return r"MSE($I_t$,$I_0$), $\Delta I_t$ [Log-scale]"
            else: raise ValueError(f"Uknown name [{name}]")
        new = edict({'x':None,'y':None})
        new.x = translate_label(labels.x)
        new.y = translate_label(labels.y)
        return new

    axis_labels = edict({'x':field1,'y':field2})
    axis_labels = translate_axis_labels(axis_labels)
    axes[0].set_xlabel(axis_labels.x,fontsize=FONTSIZE)
    axes[0].set_ylabel(axis_labels.y,fontsize=FONTSIZE)

    # -- filter ticks --
    N = np.min([len(grids[0]),MAX_LEN_TICKS])
    if N > MAX_LEN_TICKS//2: skip = 2
    else: skip = 1
    xlocs = grids[0][::skip]
    xlabels = ["%1.1e" % x for x in f1_unique[::skip]]
    N = np.min([len(grids[1]),MAX_LEN_TICKS])
    if N > MAX_LEN_TICKS//2: skip = 2
    else: skip = 1
    ylocs = grids[1][::skip]
    ylabels = ["%1.1e" % x for x in f2_unique[::skip]]

    # -- x and y ticks --
    axes[0].set_xticks(lgrids.tickmarks[field1])
    axes[0].set_xticklabels(lgrids.tickmarks_str[field1],fontsize=FONTSIZE)

    axes[0].set_yticks(lgrids.tickmarks[field2])
    axes[0].set_yticklabels(lgrids.tickmarks_str[field2],fontsize=FONTSIZE)

    # axes[0].set_xticks(xlocs)
    # axes[0].set_xticklabels(xlabels,rotation=45,ha="right")
    # axes[0].set_yticks(ylocs)
    # axes[0].set_yticklabels(ylabels,rotation=45,ha="right")

    # -- legend --
    if add_legend_bool and (ax is None):
        proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
                 for pc in cs.collections]
        mlevels_fmt = ["%0.3f" % x for x in mlevels]
        N,skip = len(proxy),8
        skim_proxy = proxy[::N//skip]
        skim_fmt = mlevels_fmt[1::N//skip] # skip 1st element; return max of interval
        if skim_fmt[-1] != mlevels_fmt[-1]: #N % skip != 0:
            skim_proxy += [proxy[-1]]
            skim_fmt += [mlevels_fmt[-1]]
        add_legend(axes[0],"Approx. Prob.",skim_fmt,skim_proxy,
                   framealpha=0.,shrink_perc=shrink_perc)

    # -- save/file io --
    # axes[1].contourf(f1_unique,f2_unique,stds.T,levels=slevels)
    # axes[1].set_title(f"Vars",fontsize=FONTSIZE)
    # axes[1].set_xlabel(f"Field [{field1}]",fontsize=FONTSIZE)
    
    if ax is None:
        DIR = Path("./output/pretty_plots")
        if not DIR.exists(): DIR.mkdir()
        fn =  DIR / f"./stat_test_properties_{fname}_contours-{field1}-{field2}.png"
        plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
        plt.close('all')
        print(f"Wrote plot to [{fn}]")

    return cs



def plot_sim_test(sims,lgrids,title,fname):


    # -- filter D to [1,2,3,4] --
    # sims = filter_pandas_by_field_grid(sims,"D",[1.,2.,3.,4.])
    # print(sims)
    # exit()
    
    # -- plot est-mean/std v.s. D --
    plot_single_sim_group(sims,lgrids,title,fname,"D",logx=True)
        
    # -- plot est-mean/std v.s. mu2 --
    plot_single_sim_group(sims,lgrids,title,fname,"mu2",logx=True)

    # -- plot est-mean/std v.s. std --
    plot_single_sim_group(sims,lgrids,title,fname,"std",logx=True)
    
    # -- plot countour individually --
    plot_sim_test_pairs(None,sims,lgrids,None,title,fname,"D","std",True)

    # -- get levels for contour --
    pairs = [["D","mu2"],["D","std"],["mu2","std"]]
    mlevels = get_agg_mlevels_from_pairs(sims,pairs)
    fig,axes = plt.subplots(ncols=3,figsize=(5*3+2,4))
    # fig,axes = None,[None,None,None]

    # -- plot D v.s. std --
    cs = plot_sim_test_pairs(axes[0],sims,lgrids,mlevels,title,fname,"D","std",False)

    # -- plot D v.s. mu2 --
    cs = plot_sim_test_pairs(axes[1],sims,lgrids,mlevels,title,fname,"D","mu2",False)

    # -- plot mu2 v.s. std --
    cs = plot_sim_test_pairs(axes[2],sims,lgrids,mlevels,title,fname,"mu2","std",True)

    # -- save aggregate --
    print(mlevels)
    save_pair_contours(axes,mlevels,fname,cs,"standard-summary")

    # -- stratify contours across ub --
    fields,sfield,zinfo = ["D","std"],["mu2"],None
    stratify_contour_plots(None,fields,sfield,sims,lgrids,mlevels,title,fname,zinfo)

    print(sims)
