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
from .standard import run_standard
from .proposed import run_proposed


def run():
    print("PID: {}".format(os.getpid()))    
    
    # run_standard()
    run_proposed()



