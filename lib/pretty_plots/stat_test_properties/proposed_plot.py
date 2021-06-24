

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
from pyutils import add_legend,create_named_meshgrid,np_log

# -- local imports --
from .plot import plot_single_sim_group

def plot_proposed_sims(sims,lgrids,title,fname):

    # -- plot est-mean/std v.s. D --
    plot_single_sim_group(sims,lgrids,title,fname,"D",logx=True)
        
    # -- plot est-mean/std v.s. mu2 --
    plot_single_sim_group(sims,lgrids,title,fname,"pmis",logx=False)

    # -- plot est-mean/std v.s. std --
    plot_single_sim_group(sims,lgrids,title,fname,"std",logx=True)

    # -- plot est-mean/std v.s. std --
    plot_single_sim_group(sims,lgrids,title,fname,"T",logx=False)


    print(sims)

