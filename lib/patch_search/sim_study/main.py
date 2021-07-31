"""

simulate results for learned function

"""

# -- python imports --
import functools
import numpy as np
import pandas as pd
import numpy.random as npr
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

# -- pytorch imports --

# -- project imports --
from settings import ROOT_PATH
from pyutils import create_subset_grids,create_named_meshgrid,apply_mesh_filters

def sim_gamma(shape,scale,size):
    return npr.gamma(shape,scale,size)

def sim_gaussian(mean,std,size):
    return npr.normal(mean,std,size)

def sim_prop_gaussian(ps,std,mu2ave,subset_size,T):
    std2 = (std/255.)**2
    std2_s = std2 / subset_size + std2 / T
    std2_s = 4 * std2_s * mu2ave
    std_s = np.sqrt(std2_s)
    return sim_gaussian(mu2ave,std_s,1)
    
def sim_prop_gamma(ps,std,subset_size,T):
    std2 = (std/255.)**2
    std_s = std2 / subset_size + std2 / T
    return sim_gamma(ps,2*std_s/ps,1)

def choose_minSN(select,nframes):
    if select == "one": minSN = 1
    elif select == "half": minSN = nframes//2
    elif select == "2/3": minSN = int(nframes * (2/3))
    else: raise ValueError(f"Uknown minSN [{select}]")
    return minSN

#@functools.lru_cache(maxsize=10)
def proposed_metric(named_mesh,nreps):
    events = []
    for m in tqdm(named_mesh):
        indices = np.arange(m.nframes)
        max_size = 600
        minSN = choose_minSN(m.minSN,m.nframes)
        subsets_idx = create_subset_grids(minSN,m.nframes-1,indices,max_size)
        vals = []
        for r in range(nreps):
            value = 0
            for subset in subsets_idx:
                subset_size = len(subset)
                simG = sim_prop_gamma(m.ps,m.std,subset_size,m.nframes)
                simN = sim_prop_gaussian(m.ps,m.std,m.mu2ave,subset_size,m.nframes)
                value += simG + simN
            vals.append(value)
        valueAve = np.mean(vals)
        valueStd = np.std(vals)
        event = {'mu2ave':m.mu2ave,'std':m.std,'nframes':m.nframes,
                     'minSN':m.minSN,'valueAve':valueAve,'valueStd':valueStd}
        events.append(event)
    events = pd.DataFrame(events)
    return events

def get_v1_mesh():
    
    # -- create grids --
    N = 20
    mu2ave_max = 1e-3
    mu2ave_grid = list(np.linspace(0,mu2ave_max,N)/mu2ave_max)
    std_grid = [25,50,75]
    ps_grid = [9,25]
    t_grid = [3,7,15]
    minSN_grid = ['one','half','2/3']

    lists = [mu2ave_grid,std_grid,ps_grid,t_grid,minSN_grid]
    order = ['mu2ave','std','ps','nframes','minSN']

    # -- create meshgrid --
    named_mesh = create_named_meshgrid(lists,order)
    filters = [{'nframes-minSN':[[3,'one'],[3,'half'],[7,'one'],[7,'half'],[7,'2/3'],
                                 [15,'one'],[15,'half'],[15,'2/3']]}]
    named_mesh = apply_mesh_filters(named_mesh,filters)
    version = "1"
    return named_mesh,version


def run():
    
    reset = False
    # -- v1 --
    named_mesh,version = get_v1_mesh()

    # -- sample RV --
    nreps = 30
    DIR = ROOT_PATH / Path(f"output/sim_study/{version}")
    if not DIR.exists(): DIR.mkdir()
    cache_fn = DIR / "cache.csv"
    if cache_fn.exists() and reset is False:
        events = pd.read_csv(cache_fn)
    else:
        events = proposed_metric(named_mesh,nreps)
        events.to_csv(cache_fn,index=False)
    print(events)

    # -- compute plots --
    compute_mu2ave_v_valueAve_plots(events,DIR)
    compute_valueStd_v_nframes_plots(events,DIR)
    
def compute_valueStd_v_mu2ave_v_nframes_plots(events):
    pass

def compute_valueStd_v_nframes_plots(events,DIR):
    
    events = events[events['minSN'] == 'half']
    for mu2ave,df in events.groupby('mu2ave'):
        nframes = df['nframes'].to_numpy()
        valueAve = df['valueAve'].to_numpy()
        valueStd = df['valueStd'].to_numpy()
        
        # -- sort --
        order = np.argsort(nframes)
        nframes = nframes[order]
        valueStd = valueStd[order]

        # -- plots --
        fig,ax = plt.subplots()
        ax.errorbar(nframes,valueStd,yerr=valueStd)
        none = "default"
        fn = DIR / f"valueStd_v_nframes_{none}.png"
        plt.savefig( fn, dpi = 300, bbox_inches='tight')
        plt.close("all")
    
def compute_mu2ave_v_valueAve_plots(events,DIR):

    # -- compute moments --
    # events = events[events['minSN'] == 'one']
    events = events[events['minSN'] == 'half']
    for nframes,df in events.groupby('nframes'):
        mu2ave = df['mu2ave'].to_numpy()
        valueAve = df['valueAve'].to_numpy()
        valueStd = df['valueStd'].to_numpy()
        
        # -- sort --
        order = np.argsort(mu2ave)
        mu2ave = mu2ave[order]
        valueAve = valueAve[order]
        valueStd = valueStd[order]

        # -- plots --
        fig,ax = plt.subplots()
        ax.errorbar(mu2ave,valueAve,yerr=valueStd)
        fn = DIR / f"mu2ave_v_valueAve_{nframes}.png"
        plt.savefig( fn, dpi = 300, bbox_inches='tight')
        plt.close("all")
