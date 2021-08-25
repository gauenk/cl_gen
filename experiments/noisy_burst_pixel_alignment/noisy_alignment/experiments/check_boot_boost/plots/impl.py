
# -- python imports --
from easydict import EasyDict as edict

# -- project imports --
from pyutils.misc import stats_by_unique

# -- local imports --
from ._lines import plot_stats
from ._stratify import stratify_line_plots
from ._translate import xfer_fields_to_labels

def plot_experiment(records,plt_fmt,exp_cfgs):
    create_2d_plots(records,plt_fmt,exp_cfgs)
    create_contour_plots(records,plt_fmt,exp_cfgs)

def create_2d_plots(records,plt_fmt,exp_cfgs):
    # -- 2d plots --
    fname = "check_boot_boost"
    info = edict({'group':'dscores','title':'Subset v.s. Mean MSE'})
    field_y = "dscores"
    field_x_list = ["nboot","nframes","std"]
    for field_x in field_x_list:
        stats = stats_by_unique(records,field_x,field_y)
        plot_stats(stats,field_x,field_y,plt_fmt,title,fname,input_ax=None)

    # -- stratify line across ub --
    fields,sfield = ["D","pmis","std","T"],["ub"]
    stratify_line_plots(records,fields,sfield,sims,plt_fmt,title,fname)

    # -- stratify line across frames --
    fields,sfield = ["D","pmis","std","ub"],["T"]
    stratify_line_plots(records,fields,sfield,sims,plt_fmt,title,fname)

def create_contour_plots(records,plt_fmt,exp_cfgs):
    # -- create pairs for contour plots --
    fields = ["D","pmis","std","T","ub"]
    pairs = create_list_pairs(fields)
    
    # -- setup zaxis vars --
    xform = None
    title = f"Approx. Probability of Correct Alignment [{fname.capitalize()}]"
    label = "Approx. Prob."
    zinfo = edict({'mean':'est_mean',
                   'std':'est_std',
                   'title':title,
                   'label':label,
                   'xform':xform})
    title = f"Approx. Mean of Subset-MSE"
    label = "Approx. Mean"
    z_mu2 = edict({'mean':'est_mu2_mean',
                       'std':'est_mu2_std',
                       'title':title,
                       'label':label,
                       'xform':xform})

    # -- create contour plots --
    ax,mlevels,legend = None,None,True
    for pair in pairs:
        f1,f2 = pair
        cs = plot_sim_test_pairs(ax,sims,plt_fmt,mlevels,title,fname,f1,f2,legend,zinfo)
        cs = plot_sim_test_pairs(ax,sims,plt_fmt,mlevels,title,fname,f1,f2,legend,z_mu2)

    # -- stratify contours across T --
    fields,sfield = ["D","pmis","std","ub"],["T"]
    stratify_contour_plots(None,fields,sfield,sims,plt_fmt,mlevels,title,fname,zinfo)

