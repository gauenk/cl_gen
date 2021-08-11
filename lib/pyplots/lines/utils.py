from .groups import plot_single_sim_group

def stratify_line_plots(ax,fields,sfields,sims,lgrids,title,fname,zinfo=None):
    fmt = format_sfields_str(sfields)
    meshgrid = create_sfields_mesh(sims,sfields)
    cs_list = []
    for elem in meshgrid:
        fsims = filter_sims(sims,elem)
        elem_val = tuple([elem[f] for f in sfields])
        fmt_val_str = fmt % elem_val
        fname_p = fname + fmt_val_str
        for field in fields:
            cs = plot_single_sim_group(ax,fsims,lgrids,title,fname_p,
                                       field,logx=True,scatter=True)
            cs_list.append(cs)
    return cs_list

