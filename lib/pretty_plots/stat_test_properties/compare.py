
# -- python --
import numpy as np
import pandas as pd
import numpy.random as npr
from easydict import EasyDict as edict

# -- python plotting --
import matplotlib
import seaborn as sns
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import cm as plt_cm

# -- project imports --
from pyutils import add_legend,create_meshgrid,np_log,create_list_pairs

# -- local imports --
from .settings import FONTSIZE
from .misc import format_sfields_str,create_sfields_mesh,filter_sims,filter_pandas_by_field_range,filter_pandas_by_field_grid
from .plot import plot_single_sim_group,stratify_contour_plots,remove_axes_labels,subplot_titles
from .plot import plot_sim_test_pairs,get_agg_mlevels_from_pairs,save_pair_contours,plot_p1boundary


def run_compare_sims(s_sims,s_lgrids,p_sims,p_lgrids):
    
    # -- compare D-std plots for standard and proposed stratified by "T" --
    compare_D_std(s_sims,s_lgrids,p_sims,p_lgrids)


def compare_D_std(s_sims,s_lgrids,p_sims,p_lgrids):
    

    # -- filter mats --
    s_sims = filter_pandas_by_field_range(s_sims,'std',0,100)
    s_sims = filter_pandas_by_field_range(s_sims,'mu2',10,20)
    s_sims = filter_pandas_by_field_grid(s_sims,'T',[3,10,50])
    p_sims = filter_pandas_by_field_range(p_sims,'std',0,100)
    p_sims = filter_pandas_by_field_range(p_sims,'ub',10,20)
    p_sims = filter_pandas_by_field_grid(p_sims,'T',[3,10,50])
    p_sims = filter_pandas_by_field_grid(p_sims,'std',p_lgrids.grids['std'])
    p_sims = filter_pandas_by_field_grid(p_sims,'D',p_lgrids.grids['D'])
    # p_sims = p_sims.sort_values("T")
    print(s_sims['mu2'].unique())
    print(p_sims['ub'].unique())
    print(p_sims.columns.to_list())
    print(p_sims['T'].unique())
    print(p_sims['std'].unique())
    
    # -- setup zaxis vars --
    def apply_search(means): return means**(10)
    xform = apply_search
    fname = "compare"
    title = f"Approx. Probability of Correct Alignment [{fname.capitalize()}]"
    label = "Approx. Prob."
    zinfo = edict({'mean':'est_mean','std':'est_std',
                   'title':title,'label':label,
                   'xform':xform})

    # -- get levels for contour color --
    pairs = [["D","std"]]
    mlevels_s = get_agg_mlevels_from_pairs(s_sims,pairs,xform)
    pairs = [["D","std"]]
    mlevels_p = get_agg_mlevels_from_pairs(p_sims,pairs,xform)
    # mlevels = np.unique(sorted(np.r_[mlevels_s,mlevels_p]))
    # mlevels = np.linspace(.4,1.,30)
    mlevels = np.linspace(.0,1.,30)
    # print(mlevels)
    # if not(xform is None): mlevels = xform(mlevels)
    # mlevels = np.array([1e-4,1e-3,1e-2,.99,1])
    # print(mlevels[::3])
    # print(mlevels.min())

    # -- # of T stratifications --
    Tgrid = sorted(p_sims['T'].unique())

    # -- create figure --
    N = len(Tgrid)
    fig,axes = plt.subplots(ncols=N,nrows=2,figsize=(4*N,3*2),
                            sharey=True,sharex=True)
    fig.subplots_adjust(hspace=0.05,wspace=0.05)
    # axes = [[axes[i]] for i in range(2*N)] # comply with stratify_contour_plots indexing

    # -- stratify contour plots --
    # title = "Standard Function"
    # fname = "compare"
    # plot_sim_test_pairs(axes[0][0],s_sims,s_lgrids,
    #                     mlevels,title,fname,"D","std",False,zinfo)

    axes_strat = [[axes[0][i]] for i in range(N)]
    fields,sfield = ["D","std"],["T"]
    title = "Standard Function"
    fname = "compare"
    cs_list = stratify_contour_plots(axes_strat,fields,sfield,s_sims,s_lgrids,
                                     mlevels,title,fname,zinfo)

    # axes_strat = axes[N:]
    axes_strat = [[axes[1][i]] for i in range(N)]
    fields,sfield = ["D","std"],["T"]
    title = "Proposed Function"
    fname = "compare"
    cs_list = stratify_contour_plots(axes_strat,fields,sfield,p_sims,p_lgrids,
                                     mlevels,title,fname,zinfo)
    print(cs_list)

    # -- plot probability 1 boundary --
    # title = "Prob 1 Boundary"
    # zinfo.ls = '--'
    # zinfo.color = 'r'
    # cs = plot_p1boundary(axes[-1][0],s_sims,s_lgrids,
    #                      title,fname,"D","std",zinfo)
    # zinfo.ls = '-'
    # zinfo.color = 'b'
    # p_sims_t3 = filter_pandas_by_field_grid(p_sims,"T",[3])
    # cs = plot_p1boundary(axes[-1][0],p_sims_t3,p_lgrids,
    #                      title,fname,"D","std",zinfo)
    # zinfo.ls = '-'
    # zinfo.color = 'k'
    # p_sims_t10 = filter_pandas_by_field_grid(p_sims,"T",[10])
    # cs = plot_p1boundary(axes[-1][0],p_sims_t10,p_lgrids,
    #                      title,fname,"D","std",zinfo)
    # cs_list.append(cs[0])

    # -- save contour plots --
    fname = "compare"
    axes = [ax[0] for ax in axes] # unlist for compat.
    print(mlevels)
    xlabel = axes[0].get_xlabel()
    print(xlabel)
    axes[0].set_xlabel("")
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both',
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel,fontsize=FONTSIZE)
    remove_axes_labels(axes[1:])
    titles = ["Standard"] + [f"\# of Frames: {T}" for T in Tgrid]
    subplot_titles(axes,titles)

    save_pair_contours(axes,mlevels,fname,cs_list[-2],"D-std-strat-T")


