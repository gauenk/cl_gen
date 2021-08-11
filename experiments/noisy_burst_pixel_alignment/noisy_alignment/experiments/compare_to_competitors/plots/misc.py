def filter_ticks(max_len,grids):
    N = np.min([len(grids[0]),max_len])
    if N > max_len//2: skip = 2
    else: skip = 1
    xlocs = grids[0][::skip]
    xlabels = ["%1.1e" % x for x in f1_unique[::skip]]
    N = np.min([len(grids[1]),max_len])
    if N > max_len//2: skip = 2
    else: skip = 1
    ylocs = grids[1][::skip]
    ylabels = ["%1.1e" % x for x in f2_unique[::skip]]
    return xlabels,ylabels

def filter_pandas_by_field_grid(sims,field,values):
    return sims.loc[sims[field].isin(values)]

def filter_pandas_by_field_range(sims,field,vmin,vmax):
    return sims[sims[field].between(vmin,vmax)]

def get_fields_fmt(field):
    if field in ["D","T","pmis"]: return "%d"
    else: return "%1.1e"

def format_sfields_str(sfields):
    fmt = "_"
    for idx,sfield in enumerate(sfields):
        fmt += f"{sfield}-"
        fmt += get_fields_fmt(sfield)
        if (idx+1) < len(sfields): fmt += "_"
    return fmt


def create_sfields_mesh(sims,sfields):
    uniques = []
    for sfield in sfields:
        uniques_f = sorted(sims[sfield].unique())
        uniques.append(uniques_f)
    mesh = create_named_meshgrid(uniques,sfields)
    return mesh

def filter_sims(sims,elem):
    fsims = sims
    for k,v in elem.items():
        fsims = fsims[fsims[k] == v]
    return fsims

def get_default_axis(ax,figsize,title):
    if ax is None:
        fig,axes = plt.subplots(figsize=figsize)
        axes = [axes]
        axes[0].set_title(title,fontsize=FONTSIZE)
    else:
        axes = [ax]
    return axes

