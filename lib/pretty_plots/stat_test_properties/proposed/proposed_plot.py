

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
from pretty_plots.stat_test_properties.misc import skip_with_endpoints,filter_sim_fields_range
from pretty_plots.stat_test_properties.plot import plot_single_sim_group,plot_sim_test_pairs,stratify_contour_plots,stratify_line_plots


def plot_proposed_sims(sims,lgrids,title,fname):

    # -- filter std to a practical range --
    # sims = filter_sim_fields_range(sims,lgrids,"std",0,100)

    # -- 2d plots --
    plot_2d = False
    if plot_2d:
        mu2_info = edict({'group':'est_mu2','title':'Subset v.s. Mean MSE'})
        fields = ["D","pmis","std","T","ub"]
        for field in fields:
            plot_single_sim_group(sims,lgrids,title,fname,field,logx=True,scatter=True)
            plot_single_sim_group(sims,lgrids,title,fname,field,mu2_info,
                                  logx=True,scatter=True)
    
        # -- stratify line across ub --
        fields,sfield = ["D","pmis","std","T"],["ub"]
        stratify_line_plots(fields,sfield,sims,lgrids,title,fname)
    
        # -- stratify line across frames --
        fields,sfield = ["D","pmis","std","ub"],["T"]
        stratify_line_plots(fields,sfield,sims,lgrids,title,fname)

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
        cs = plot_sim_test_pairs(ax,sims,lgrids,mlevels,title,fname,f1,f2,legend,zinfo)
        cs = plot_sim_test_pairs(ax,sims,lgrids,mlevels,title,fname,f1,f2,legend,z_mu2)

    # -- stratify contours across T --
    fields,sfield = ["D","pmis","std","ub"],["T"]
    stratify_contour_plots(None,fields,sfield,sims,lgrids,mlevels,title,fname,zinfo)

    # -- stratify contours across ub --
    fields,sfield = ["D","pmis","std","T"],["ub"]
    stratify_contour_plots(None,fields,sfield,sims,lgrids,mlevels,title,fname,zinfo)

    # -- stratify contours across ub & D --
    fields,sfield = ["std","T"],["ub","D"]
    stratify_contour_plots(None,fields,sfield,sims,lgrids,mlevels,title,fname,zinfo)

    # -- stratify contours across ub & T --
    fields,sfield = ["std","D"],["ub","T"]
    stratify_contour_plots(None,fields,sfield,sims,lgrids,mlevels,title,fname,zinfo)

    print(sims)

