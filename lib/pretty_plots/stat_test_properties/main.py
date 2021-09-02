# -- python imports --
import os
import math
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as npr
from easydict import EasyDict as edict
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from matplotlib import cm as plt_cm
from joblib import Parallel,delayed

# -- project imports --
from pyutils import add_legend,create_named_meshgrid,np_log

# -- [local] project imports --
from .standard import run_standard,get_standard_sims,create_standard_parameter_grid
from .proposed import run_proposed,get_proposed_sims,create_proposed_parameter_grid
from .hb_gaussian import run_hb_gaussian
from .compare import run_compare_sims


def run():
    print("PID: {}".format(os.getpid()))    
    
    # -- run single experiment --
    # run_standard()
    # run_proposed()
    # run_hb_gaussian()

    # -- compare standard and proposed --
    pgrid,s_lgrids = create_standard_parameter_grid()
    s_sims = get_standard_sims(pgrid)
    pgrid,p_lgrids = create_proposed_parameter_grid()
    p_sims = get_proposed_sims(pgrid)
    run_compare_sims(s_sims,s_lgrids,p_sims,p_lgrids)


