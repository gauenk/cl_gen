
# -- python imports --
import numpy as np
import numpy_groupies as npg
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

# -- project imports --
from pyutils import create_named_meshgrid,np_log,add_legend

# -- local imports --
from .settings import FONTSIZE,MAX_LEN_XLABEL

def skip_with_endpoints(ndarray,skip):
    skipped = ndarray[::skip]
    skipped[-1] = ndarray[-1]
    return skipped

def add_contour_legend(axes,ax,cs,add_legend_bool,mlevels,zinfo,shrink_perc):
    if add_legend_bool and (ax is None):
        proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
                 for pc in cs.collections]
        mlevels_fmt = ["%0.3f" % x for x in mlevels]
        N,num = len(proxy),8
        skip = N//num
        if skip < 1: skip = 1
        skim_proxy = proxy[::skip]
        skim_fmt = mlevels_fmt[1::skip] # skip 1st element; return max of interval
        skim_ml = mlevels[1::skip] # skip 1st element; return max of interval
        if skim_fmt[-1] != mlevels_fmt[-1]: # N % skip != 0:
            skim_proxy[-1] = proxy[-1]
            skim_fmt[-1] = mlevels_fmt[-1]
            skim_ml[-1] = mlevels[-1]
        add_legend(axes[0],zinfo.label,skim_fmt,skim_proxy,
                   framealpha=0.,shrink_perc=shrink_perc)


def get_color_range(mlevels,means,vmin,vmax):
    if mlevels is None:
        if vmin is None: vmin = means.min()
        if vmax is None: vmax = means.max()
    else: vmin,vmax = np.min(mlevels),np.max(mlevels)
    return vmin,vmax

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

def compute_max_yval_boundary_rc(means,val=0.9):
    #rows,cols = np.where(np.isclose(means,val))
    rows,cols = np.where(means>val)
    order = np.argsort(cols)
    cols = cols[order]
    rows = rows[order]
    rows = npg.aggregate(cols,rows,func='max')
    cols = np.unique(cols)
    return rows,cols    

def compute_max_yval_boundary(means,grids,val=0.9):
    rows,cols = compute_max_yval_boundary_rc(means,val=val)
    print(rows,cols)
    y_values = np.array(grids[1])[rows]
    x_values = np.array(grids[0])[cols]
    return x_values,y_values

def get_default_zinfo(zinfo,fname):
    if zinfo is None:
        t_fname = fname
        if "_" in t_fname: t_fname = fname.split("_")[0]
        title = f"Approx. Probability of Correct Alignment [{t_fname.capitalize()}]"
        label = "Approx. Prob."
        zinfo = edict({'mean':'est_mean','std':'est_std'})
        zinfo.title = title
        zinfo.label = label
        zinfo.ls = '-'
        zinfo.color = 'k'
    return zinfo

def translate_axis_labels(labels,logs):
    def translate_label(name):
        if name == "D": return "Patchsize, $D$"
        elif name == "std": return "Noise Level, $\sigma^2$"
        elif name == "mu2": return r"MSE($I_t$,$I_0$), $\Delta I_t$"
        elif name == "pmis": return r"Percent of Misaligned Frames"
        elif name == "ub": return r"Upper Bound of Misaligned Pixel Value"
        elif name == "T": return r"Number of Frames" 
        elif name == "est_mu2_mean": return r"Est. Mean of MSE Between Subset Aves."
        elif name == "eps": return r"Gaussian Alignment Variance"        
        else: raise ValueError(f"Uknown name [{name}]")
    new = edict({'x':None,'y':None})
    new.x = translate_label(labels.x)
    new.y = translate_label(labels.y)
    if logs.x: new.x += " [Log Scale]"
    if logs.y: new.y += " [Log Scale]"
    return new


def compute_log_lists(grids,fields,logs):
    for idx,field in enumerate(fields):
        if logs[field]:
            grids[idx] = np_log(grids[idx])/np_log([10])
    return grids

def compute_stat_contour(sims,fgrid,grids,fields,zinfo):
    means = np.zeros((len(grids[1]),len(grids[0])))
    stds = np.zeros((len(grids[1]),len(grids[0])))
    for point in fgrid:
        # -- filter fields --
        filter1 = sims[sims[fields[0]] == point.f1]
        filter2 = filter1[filter1[fields[1]] == point.f2]

        # -- estimate parameters --
        est_mean = np.mean(np.stack(filter2[zinfo.mean].to_numpy()).flatten())
        est_std = np.std(np.stack(filter2[zinfo.std].to_numpy()).flatten())

        std = np.unique(filter2['std'].to_numpy())
        eps = np.unique(filter2['eps'].to_numpy())
        T = np.unique(filter2['T'].to_numpy())
        D = np.unique(filter2['D'].to_numpy())
        gt = std**2 / T + eps**2 / T
        error = (est_mean - gt)**2
        #print("%2.3e"%(error),est_mean,gt,std,eps,T,D)
        est_mean = error

        # -- format samples for contour plot --
        f1_index = grids[0].index(point.f1)
        f2_index = grids[1].index(point.f2)
        means[f2_index,f1_index] = est_mean
        stds[f2_index,f1_index] = est_std

    return means,stds

def aggregate_field_pairs(sims,fgrid,grids,fields,zinfo):
    agg,xdata,ydata,gt = [],[],[],[]
    for point in fgrid:
        # -- filter fields --
        filter1 = sims[sims[fields[0]] == point.f1]
        filter2 = filter1[filter1[fields[1]] == point.f2]

        # -- estimate parameters --
        est_mean = np.mean(np.stack(filter2[zinfo.mean].to_numpy()).flatten())
        est_std = np.std(np.stack(filter2[zinfo.std].to_numpy()).flatten())

        std = np.unique(filter2['std'].to_numpy())
        eps = np.unique(filter2['eps'].to_numpy())
        T = np.unique(filter2['T'].to_numpy())
        D = np.unique(filter2['D'].to_numpy())
        gt = std**2 / T + eps**2 / T

        # -- aggregate data --
        xdata.append([std,eps,T,D])
        ydata.append([est])
        gt.append([gt])
        agg.append({'gt':gt,'std':std,'eps':eps,'T':T,'D':D,'est':est})

    return agg,xdata,ydata,gt

def filter_sim_fields_range(sims,lgrid,field,lb,ub):
    # -- filter data --
    fsims = sims[sims[field].between(lb,ub)]
    use_log = lgrid.logs[field]

    # -- modify plot info --
    if use_log: lb,ub = np_log(lb),np_log(ub)
    start,end,base = lb,ub,10
    size = len(lgrid.tickmarks[field])
    tickmarks = np.linspace(start,end,end - start + 1)
    tickmarks = skip_with_endpoints(tickmarks,2)
    if use_log: ticks = np.power([base],tickmarks)
    else: ticks = tickmarks
    tickmarks_str = ["%d" % x for x in tickmarks]

    
    lgrid.ticks[field] = ticks
    lgrid.tickmarks[field] = tickmarks
    lgrid.tickmarks_str[field] = tickmarks_str

    return fsims

