
# -- python imports --
from easydict import EasyDict as edict

# -- project imports --
from pyutils import np_log,create_named_meshgrid,create_list_pairs
from pyplots.legend import add_legend,add_colorbar
from pyplots.misc import order_field_pairs,get_default_zinfo,compute_stat_contour


def plot_contour_pairs(ax,sims,lgrids,mlevels,title,fname,field1,field2,
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

