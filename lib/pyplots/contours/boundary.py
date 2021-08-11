

# -- project imports --
from pyutils import np_log,create_named_meshgrid,create_list_pairs
from pyplots.legend import add_legend,add_colorbar
from pyplots.misc import order_field_pairs


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


