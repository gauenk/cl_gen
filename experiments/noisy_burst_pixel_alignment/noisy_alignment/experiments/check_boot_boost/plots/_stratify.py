
from ._lines import plot_stats

def stratify_line_plots(axes,fields,sfields,sims,plt_fmt,mlevels,title,fname,zinfo):
    raise NotImplemented("")
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
            cs = plot_sim_test_pairs(ax,fsims,plt_fmt,mlevels,title,
                                     fname_p,f1,f2,legend,zinfo)
            cs_list.append(cs)
    return cs_list

