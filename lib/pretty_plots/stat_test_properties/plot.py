# -- python --
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
from einops import repeat

# -- python plotting --
import matplotlib
import seaborn as sns
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import cm as plt_cm

# -- project imports --
from pyutils import create_named_meshgrid,np_log,create_list_pairs
from pyplots import add_legend,add_colorbar

# -- local imports --
from .settings import FONTSIZE,MAX_LEN_XLABEL
from .misc import format_sfields_str,create_sfields_mesh,filter_sims,get_default_zinfo,compute_stat_contour,translate_axis_labels,get_color_range,add_contour_legend,compute_log_lists,compute_max_yval_boundary,get_default_axis

def add_jitter(ndarray,std=0.05):
    return np.random.normal(ndarray,scale=std)

# -=-=-=-=-=-=-=-=-=-
#
#
#
#
# ---- 2d Plots ----
#
#
#
#
# -=-=-=-=-=-=-=-=-=-

def plot_single_sim_group(sims,lgrids,title,fname,group,yinfo=None,
                          logx=True,scatter=False):
    
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
    fig,ax = plt.subplots(figsize=(8,4))
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
    DIR = Path(f"./output/pretty_plots/stat_test_properties/{fname}/")
    if not DIR.exists(): DIR.mkdir()
    fn =  DIR / f"./stat_test_properties_{fname}_group-{group}-{yinfo.group}.png"
    plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")
    plt.xscale("linear")

# -=-=-=-=-=-=-=-=-=-=-=-
#
#
#
#
# ---- Contour Maps ----
#
#
#
#
# -=-=-=-=-=-=-=-=-=-=-=-

def order_field_pairs(field1,field2):
    if field1 == "std" and field2 == "D":
        return "D","std"
    else:
        return field1,field2

def plot_sim_test_pairs(ax,sims,lgrids,mlevels,title,fname,field1,field2,
                        add_legend_bool=True,zinfo=None,vmin=None,vmax=None):
    
    MAX_LEN_TICKS = 30
    shrink_perc = 0.8

    # -- order field pairs for consistency --
    field1,field2 = order_field_pairs(field1,field2)
    fields = [field1,field2]

    # -- default z info --
    zinfo = get_default_zinfo(zinfo,fname)

    # -- create field grid --
    f1_unique = sorted(list(sims[field1].unique()))
    f2_unique = sorted(list(sims[field2].unique()))
    grids = [f1_unique,f2_unique]
    names = ["f1","f2"]
    fgrid = create_named_meshgrid(grids,names)
    
    # -- compute contour map --
    means,stds = compute_stat_contour(sims,fgrid,grids,fields,zinfo)
    if not(zinfo.xform is None): means = zinfo.xform(means)
    
    # -- number of contour levels --
    if mlevels is None:
        quants = np.linspace(0.00,1.0,30)
        mlevels = np.linspace(means.min(),means.max(),30)
        # mlevels = np.unique(np.quantile(means,quants))
    
    # -- log axis --
    grids = compute_log_lists(grids,fields,lgrids.logs)

    # -- figure size depends on legend inclusion --
    figsize = [6,4]
    # if add_legend_bool and (ax is None): figsize[0] = figsize[0]/shrink_perc

    # -- plot contours --
    title = zinfo.title
    if ax is None:
        fig,axes = plt.subplots(figsize=figsize)
        axes = [axes]
        axes[0].set_title(title,fontsize=FONTSIZE)
    else:
        axes = [ax]

    # -- color range --
    vmin,vmax = get_color_range(mlevels,means,vmin,vmax)
    # print("means",np.min(means),np.min(mlevels),np.max(means))

    # -- create the plot --
    cs = axes[0].contourf(grids[0],grids[1],means,levels=mlevels,
                          cmap="plasma",vmin=vmin,vmax=vmax)

    # -- axis labels --
    axis_labels = edict({'x':field1,'y':field2})
    axis_logs = edict({'x':lgrids.logs[field1],'y':lgrids.logs[field2]})
    axis_labels = translate_axis_labels(axis_labels,axis_logs)
    axes[0].set_xlabel(axis_labels.x,fontsize=FONTSIZE)
    axes[0].set_ylabel(axis_labels.y,fontsize=FONTSIZE)

    # -- filter ticks --
    # xticklabels,yticklabels = filter_ticks(MAX_LEN_TICKS,grids)

    # -- x and y ticks --
    axes[0].set_xticks(lgrids.tickmarks[field1])
    axes[0].set_xticklabels(lgrids.tickmarks_str[field1],fontsize=FONTSIZE)

    axes[0].set_yticks(lgrids.tickmarks[field2])
    axes[0].set_yticklabels(lgrids.tickmarks_str[field2],fontsize=FONTSIZE)


    # -- legend --
    add_contour_legend(axes,ax,cs,add_legend_bool,mlevels,zinfo,shrink_perc)

    # -- save/file io --
    # axes[1].contourf(f1_unique,f2_unique,stds.T,levels=slevels)
    # axes[1].set_title(f"Vars",fontsize=FONTSIZE)
    # axes[1].set_xlabel(f"Field [{field1}]",fontsize=FONTSIZE)
    
    if ax is None:
        DIR = Path(f"./output/pretty_plots/stat_test_properties/{fname}/")
        if not DIR.exists(): DIR.mkdir()
        fn =  DIR / f"./contours-{field1}-{field2}-{zinfo.mean}.png"
        plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
        plt.close('all')
        print(f"Wrote plot to [{fn}]")

    return cs


def stratify_line_plots(fields,sfields,sims,lgrids,title,fname,zinfo=None):
    fmt = format_sfields_str(sfields)
    meshgrid = create_sfields_mesh(sims,sfields)
    cs_list = []
    for elem in meshgrid:
        fsims = filter_sims(sims,elem)
        elem_val = tuple([elem[f] for f in sfields])
        fmt_val_str = fmt % elem_val
        fname_p = fname + fmt_val_str
        for field in fields:
            cs = plot_single_sim_group(fsims,lgrids,title,fname_p,
                                       field,logx=True,scatter=True)
            cs_list.append(cs)
    return cs_list


def stratify_contour_plots(axes,fields,sfields,sims,lgrids,mlevels,title,fname,zinfo):
    ax = None
    legend = True
    fmt = format_sfields_str(sfields)
    meshgrid = create_sfields_mesh(sims,sfields)
    pairs = create_list_pairs(fields)
    cs_list = []
    for ix,elem in enumerate(meshgrid):
        fsims = filter_sims(sims,elem)        
        elem_val = tuple([elem[f] for f in sfields])
        fmt_val_str = fmt % elem_val
        fname_p = fname + fmt_val_str
        for jx,pair in enumerate(pairs):
            if not(axes is None): ax = axes[ix][jx]
            f1,f2 = pair
            cs = plot_sim_test_pairs(ax,fsims,lgrids,mlevels,title,
                                     fname_p,f1,f2,legend,zinfo)
            cs_list.append(cs)
    return cs_list

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        Draw Max "1" Line
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def plot_p1boundary(ax,sims,lgrids,title,fname,field1,field2,zinfo=None):

    MAX_LEN_TICKS = 30
    shrink_perc = 0.8

    # -- order field pairs for consistency --
    field1,field2 = order_field_pairs(field1,field2)
    fields = [field1,field2]

    # -- default z info --
    zinfo = get_default_zinfo(zinfo,fname)

    # -- create field grid --
    f1_unique = sorted(list(sims[field1].unique()))
    f2_unique = sorted(list(sims[field2].unique()))
    grids = [f1_unique,f2_unique]
    names = ["f1","f2"]
    fgrid = create_named_meshgrid(grids,names)
    
    # -- compute contour map --
    means,stds = compute_stat_contour(sims,fgrid,grids,fields,zinfo)
    if not(zinfo.xform is None): means = zinfo.xform(means)

    # -- log axis --
    grids = compute_log_lists(grids,fields,lgrids.logs)

    # -- compute the boundary --
    rows,cols = compute_max_yval_boundary(means,grids)
    # rc_tuple = compute_log_lists([rows,cols],fields,lgrids.logs)
    # rows,cols = rc_tuple[0],rc_tuple[1]

    # -- figure size depends on legend inclusion --
    figsize,title = [6,4],zinfo.title
    # if add_legend_bool and (ax is None): figsize[0] = figsize[0]/shrink_perc
    axes = get_default_axis(ax,figsize,title)

    # -- create the plot --
    cs = axes[0].plot(rows,cols,ls=zinfo.ls,color=zinfo.color)

    # -- axis labels --
    axis_labels = edict({'x':field1,'y':field2})
    axis_logs = edict({'x':lgrids.logs[field1],'y':lgrids.logs[field1]})
    axis_labels = translate_axis_labels(axis_labels,axis_logs)
    axes[0].set_xlabel(axis_labels.x,fontsize=FONTSIZE)
    axes[0].set_ylabel(axis_labels.y,fontsize=FONTSIZE)

    # -- filter ticks --
    # xticklabels,yticklabels = filter_ticks(MAX_LEN_TICKS,grids)

    # -- x and y ticks --
    axes[0].set_xticks(lgrids.tickmarks[field1])
    axes[0].set_xticklabels(lgrids.tickmarks_str[field1],fontsize=FONTSIZE)

    axes[0].set_yticks(lgrids.tickmarks[field2])
    axes[0].set_yticklabels(lgrids.tickmarks_str[field2],fontsize=FONTSIZE)

    # -- legend --
    # add_contour_legend(axes,ax,cs,add_legend_bool,mlevels,zinfo,shrink_perc)

    # -- save/file io --
    # axes[1].contourf(f1_unique,f2_unique,stds.T,levels=slevels)
    # axes[1].set_title(f"Vars",fontsize=FONTSIZE)
    # axes[1].set_xlabel(f"Field [{field1}]",fontsize=FONTSIZE)
    
    if ax is None:
        DIR = Path(f"./output/pretty_plots/stat_test_properties/{fname}/")
        if not DIR.exists(): DIR.mkdir()
        fn =  DIR / f"./stat_test_properties_{fname}_p1boundary-{field1}-{field2}-{zinfo.mean}.png"
        plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
        plt.close('all')
        print(f"Wrote plot to [{fn}]")

    return cs


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        Save Groups of Plots
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_agg_mlevels_from_pairs(sims,pairs,xform):
    
    def get_pair_mlevel(sims,field1,field2,xform=None):

        # -- create field grid --
        f1_unique = sorted(list(sims[field1].unique()))
        f2_unique = sorted(list(sims[field2].unique()))
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
            if np.isnan(est_mean):
                print(len(filter1),point.f1)
                print(len(filter2),point.f2)
                print("est_mean",est_mean)
            f1_index = f1_unique.index(point.f1)
            f2_index = f2_unique.index(point.f2)
            means[f2_index,f1_index] = est_mean
        
        if not(xform is None): means = xform(means)
        # -- number of contour levels --
        # quants = np.linspace(0.00,1.0,30)
        # mlevels = np.unique(np.quantile(means,quants))
        vmin = means.min()
        mlevels = np.linspace(vmin,1.0,30)
        print("mlevels",means.min(),np.min(mlevels),vmin)
        return mlevels
    
    mlevels = np.array([])
    for pair in pairs:
        mlevels_pair = get_pair_mlevel(sims,pair[0],pair[1],xform)
        mlevels = np.r_[mlevels,mlevels_pair]
    mlevels = np.unique(mlevels)
    # mlevels = np.unique(np.r_[mlevels_D_mu2,mlevels_D_std,mlevels_mu2_std])

    return mlevels

def remove_axes_labels(axes):
    for ax in axes:
        if isinstance(ax,list) or isinstance(ax,np.ndarray):
            remove_axes_labels(ax)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")

def subplot_titles(axes,titles):
    for ax,title in zip(axes,titles):
        ax.set_title(title,fontsize=FONTSIZE)

def make_space_above(axes, topmargin=1):
    """ increase figure size to make topmargin (in inches) space for 
        titles, without changing the axes sizes"""
    fig = axes[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

def save_pair_contours(axes,mlevels,fname,cs,postfix):
    if axes[0] is None: return
    
    # -- add legend --
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
             for pc in cs.collections]
    mlevels_fmt = ["%0.2f" % x for x in mlevels]
    N,num = len(proxy),5
    skip = N//num
    if skip < 1: skip = 1
    skim_proxy = proxy[::skip]
    skim_fmt = mlevels_fmt[1::skip]
    skim_ticks = mlevels[1::skip]
    if mlevels_fmt[-1] != skim_fmt[-1]:
        skim_proxy[-1] = proxy[-1]
        skim_fmt[-1] = mlevels_fmt[-1]
        skim_ticks[-1] = mlevels[-1]
    print(axes[-1])

    zinfo = edict({'label':'Est. Prob.'})
    # add_contour_legend([axes[-1]],None,cs,True,mlevels,zinfo,0.80)
    # add_legend(axes[3],zinfo.label,skim_fmt,skim_proxy,
    #            framealpha=0.,shrink_perc=1.)
    legend_title = zinfo.label
    legend_handles = skim_proxy
    legend_str = skim_fmt
    fontsize=15
    ncol=1
    framealpha=0.
    leg =axes[3].legend(legend_handles,legend_str,
                        title = legend_title,
                        title_fontsize=fontsize,
                        fontsize=fontsize,
                        ncol=ncol,
                        # loc='center left',
                        bbox_to_anchor=(3.55, 2.35),
                        framealpha=framealpha)


    # -- shrink non-legend far right-bottom plot --
    shrink = False
    if shrink:
        shrink_perc = .1
        box = axes[-1].get_position()
        sbox = [box.x0, box.y0, box.width * shrink_perc, box.height]
        axes[-1].set_position(sbox)

    # add_legend(axes[-1],"Approx. Prob.",skim_fmt,
    #            skim_proxy,framealpha=0.,shrink_perc=.80,
    #            fontsize=FONTSIZE)
    # print(skim_proxy)
    # print(skim_fmt)
    # label = "Est. Prob."
    # add_colorbar(axes,skim_fmt,skim_ticks,
    #              scm=skim_proxy,shrink = True,fontsize=15,
    #              framealpha=1.0,ncol=1,shrink_perc=.80)

    # -- create plots --
    # plt.subplots_adjust(right=.85)
    title = "Contour Maps of the Approximate Probability of Correct Alignment"
    # make_space_above(axes, topmargin=0.7)
    plt.suptitle(title,fontsize=18,y=0.98)

    DIR = Path("./output/pretty_plots")
    if not DIR.exists(): DIR.mkdir()
    fn =  DIR / f"./stat_test_properties_{fname}_contours-agg_{postfix}.png"
    # plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    plt.savefig(fn,transparent=True,dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")

