# -- python imports --
import os
from numba import jit
import math
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as npr
from easydict import EasyDict as edict
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm as plt_cm
from joblib import Parallel,delayed

# -- project imports --
from pyutils import add_legend,create_named_meshgrid,np_log,numba_unique,write_pickle,read_pickle

# -- local imports --
from .misc import skip_with_endpoints
from .parallel import ProgressParallel
from .plot import plot_single_sim_group

FONTSIZE = 15


def run_standard():
    pgrid,lgrid = create_standard_parameter_grid()
    plot_standard_test(pgrid,lgrid)

def plot_standard_test(pgrid,lgrid):
    sims = sim_standard_test_data(pgrid,parallel=True)
    title = "Standard Function"
    fname = "standard"
    plot_sim_test(sims,lgrid,title,fname)

def create_standard_parameter_grid():
    pgrid = edict()

    start,end,size,base = 1,20,15,10
    D_exp = np.linspace(start,end,size)
    D = np.power([base],D_exp).astype(np.long) # np.linspace(3,128,10)**2
    D_tickmarks = np.linspace(start,end,end-start+1)
    D_tickmarks = skip_with_endpoints(D_tickmarks,3)
    D_ticks = np.power([base],D_tickmarks)
    D_tickmarks_str = ["%d" % x for x in D_tickmarks]

    start,end,size,base = -15,1,50,10
    mu2_exp = np.linspace(start,end,size)
    mu2 = np.power([base],mu2_exp) # [0,1e-6,1e-3,1e-1,1,1e1]
    mu2_tickmarks = np.linspace(start,end,end - start + 1)
    mu2_tickmarks = skip_with_endpoints(mu2_tickmarks,3)
    mu2_ticks = np.power([base],mu2_tickmarks)
    mu2_tickmarks_str = ["%d" % x for x in mu2_tickmarks]

    start,end,size,base = -5,2,50,10
    std_exp = np.linspace(start,end,size)
    std = np.power([base],std_exp) # [1e-6,1e-3,1e-1,1,1e1]
    std_tickmarks = np.linspace(start,end,end - start + 1)
    std_tickmarks = skip_with_endpoints(std_tickmarks,2)
    std_ticks = np.power([base],std_tickmarks)
    std_tickmarks_str = ["%d" % x for x in std_tickmarks]

    size = [5000]

    params = [D,mu2,std,size]
    names = ['D','mu2','std','size']
    pgrid = create_named_meshgrid(params,names)
    print(f"Mesh grid created of size [{len(pgrid)}]")

    ticks = edict({'D':D_ticks,'mu2':mu2_ticks,'std':std_ticks})
    tickmarks = edict({'D':D_tickmarks,'mu2':mu2_tickmarks,'std':std_tickmarks})
    tickmarks_str = edict({'D':D_tickmarks_str,
                           'mu2':mu2_tickmarks_str,'std':std_tickmarks_str})
    # log_tickmarks = edict({'D':np_log(D_ticks),
    #                        'mu2':np_log(mu2_ticks),
    #                        'std':np_log(std_ticks)})
    lgrid = edict({'ticks':ticks,'tickmarks':tickmarks,
                   'tickmarks_str':tickmarks_str})
                   #'log_tickmarks':log_tickmarks})

    return pgrid,lgrid



def sim_standard_test_data(pgrid,parallel=True):

    def sim_standard_single(D,mu2,std,size):

        def sim_v1(D,mu2,std,size):

            gaussian_std = math.sqrt(2 * 4 * mu2 * std**2)/D
            x = npr.normal(loc=mu2,scale=gaussian_std,size=size)
    
            gamma_shape = D/2
            gamma_scale = 2*2*(std**2)/D
            y = npr.gamma(gamma_shape,scale=gamma_scale,size=size)
            z = npr.gamma(gamma_shape,scale=gamma_scale,size=size)

            left = z
            right = y + x
            cond = left < right
            return cond

        def sim_v2(D,mu2,std,size):
            D = int(D)

            left = npr.normal(loc=0,scale=std,size=(D,size))**2
            left = np.mean(left,axis=0)

            right = npr.normal(loc=math.sqrt(mu2),scale=std,size=(D,size))**2
            right = np.mean(right,axis=0)

            cond = left < right
            return cond
    
        cond = sim_v1(D,mu2,std,size)
        sim = edict()
        sim.est_mean = np.mean(cond)
        sim.est_std = np.std(cond)
    
        # -- include parameters --
        sim.mu2 = mu2
        sim.std = std
        sim.D = D
        sim = dict(sim)

        return sim
    
    
    if parallel:
        # pParallel = Parallel(n_jobs=8)
        pParallel = ProgressParallel(True,len(pgrid),n_jobs=8)
        sims = pParallel(delayed(sim_standard_single)(p.D,p.mu2,p.std,p.size)
                                   for p in pgrid)
    else:
        sims = []
        for p in pgrid:
            sims.append(sim_standard_single(p.D,p.mu2,p.std,p.size))
    sims = pd.DataFrame(sims)
    return sims
    
def plot_sim_test_pairs(ax,sims,lgrids,mlevels,title,fname,field1,field2,
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
        quants = np.linspace(0.00,1.0,30)
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
        print(len(proxy),len(mlevels_fmt))
        N,skip = len(proxy),8
        skim_proxy = proxy[::N//skip]
        skim_fmt = mlevels_fmt[1::N//skip] # skip 1st element; return max of interval
        if N % skip != 0:
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


    # -- plot est-mean/std v.s. D --
    plot_single_sim_group(sims,lgrids,title,fname,"D",logx=True)
        
    # -- plot est-mean/std v.s. mu2 --
    plot_single_sim_group(sims,lgrids,title,fname,"mu2",logx=True)

    # -- plot est-mean/std v.s. std --
    plot_single_sim_group(sims,lgrids,title,fname,"std",logx=True)
    
    # -- get levels for contour --
    mlevels = get_agg_mlevels_from_pairs(sims)
    fig,axes = plt.subplots(ncols=3,figsize=(5*3+2,4))
    # fig,axes = None,[None,None,None]

    # -- plot D v.s. std --
    cs = plot_sim_test_pairs(axes[0],sims,lgrids,mlevels,title,fname,"D","std",False)

    # -- plot D v.s. mu2 --
    cs = plot_sim_test_pairs(axes[1],sims,lgrids,mlevels,title,fname,"D","mu2",False)

    # -- plot mu2 v.s. std --
    cs = plot_sim_test_pairs(axes[2],sims,lgrids,mlevels,title,fname,"mu2","std",True)

    # -- save aggregate --
    save_pair_contours(axes,mlevels,fname,cs)

    print(sims)

def save_pair_contours(axes,mlevels,fname,cs):
    if axes[0] is None: return
    
    # -- add legend --
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
             for pc in cs.collections]
    mlevels_fmt = ["%0.3f" % x for x in mlevels]
    N,skip = len(proxy),5
    skim_proxy = proxy[::N//skip]
    skim_fmt = mlevels_fmt[1::N//skip]
    if mlevels_fmt[-1] != skim_fmt[-1]:
        skim_proxy += [proxy[-1]]
        skim_fmt += [mlevels_fmt[-1]]
    print(skim_proxy,skim_fmt)
    add_legend(axes[-1],"Approx. Prob.",skim_fmt,
               skim_proxy,framealpha=0.,shrink_perc=1.0,
               fontsize=FONTSIZE)
    plt.subplots_adjust(right=.85)
    title = "Contour Maps of the Approximate Probability of Correct Alignment"
    plt.suptitle(title,fontsize=18)

    DIR = Path("./output/pretty_plots")
    if not DIR.exists(): DIR.mkdir()
    fn =  DIR / f"./stat_test_properties_{fname}_contours-agg.png"
    plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")

def get_agg_mlevels_from_pairs(sims):
    
    def get_pair_mlevel(sims,field1,field2):

        # -- create field grid --
        f1_unique = list(sims[field1].unique())
        f2_unique = list(sims[field2].unique())
        uniques = [f1_unique,f2_unique]
        names = ["f1","f2"]
        fgrid = create_named_meshgrid(uniques,names)

        # -- aggregate over field --
        means = np.zeros((len(f2_unique),len(f1_unique)))
        for point in fgrid:
            filter1 = sims[sims[field1] == point.f1]
            filter2 = filter1[filter1[field2] == point.f2]

            # -- format samples for contour plot --
            est_mean = filter2['est_mean'].mean()
            f1_index = f1_unique.index(point.f1)
            f2_index = f2_unique.index(point.f2)
            means[f2_index,f1_index] = est_mean
        
        # -- number of contour levels --
        quants = np.linspace(0.00,1.0,30)
        mlevels = np.unique(np.quantile(means,quants))
        return mlevels
    
    mlevels_D_mu2 = get_pair_mlevel(sims,"D","mu2")
    mlevels_D_std = get_pair_mlevel(sims,"D","std")
    mlevels_mu2_std = get_pair_mlevel(sims,"mu2","std")
    
    mlevels = np.unique(np.r_[mlevels_D_mu2,mlevels_D_std,mlevels_mu2_std])
    return mlevels
