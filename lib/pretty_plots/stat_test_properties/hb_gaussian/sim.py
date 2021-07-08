
# -- python imports --
import copy,sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy.random as npr
from joblib import delayed
from easydict import EasyDict as edict

# -- project imports --
from pretty_plots.stat_test_properties.cache import store_cache,load_cache,get_cache_name,filter_complete_exps,filter_grid_field
from pretty_plots.stat_test_properties.parallel import ProgressParallel
from pretty_plots.stat_test_properties.hb_gaussian.sim_fxn import sim_bootstrap,sim_frame_v_frame,sim_frame_v_mean

def run_sim(grid,info):
    bootstrap_sims = run_bootstrap_sim(grid,parallel=True,rerun=False)
    fvf_sims = run_frame_v_frame_sim(grid,parallel=True,rerun=False)
    fvm_sims = run_frame_v_mean_sim(grid,parallel=True,rerun=False)
    sims = edict({'bs':bootstrap_sims,'fvf':fvf_sims,'fvm':fvm_sims})
    return sims

def run_frame_v_frame_sim(grid,parallel=True,rerun=False):
    cache_name = f"hb_gaussian_fvf"
    lsims = load_cache(cache_name)

    # f_grid = []
    # f_grid = grid
    grid = filter_grid_field(grid,'B')
    f_grid = filter_complete_exps(grid,lsims)
    if rerun is False and len(f_grid) == 0: return lsims
    print(f"Simulating [{len(f_grid)}] Experimental Setups with Cache [{cache_name}]")

    if parallel:
        pParallel = ProgressParallel(True,len(f_grid),n_jobs=8)
        delayed_fxn = delayed(sim_frame_v_frame)
        sims = pParallel(delayed_fxn(p.eps,p.std,p.D,p.T,p.size)
                         for p in f_grid)
    else:
        sims = []
        for p in f_grid:
            sims.append(sim_frame_v_frame(p.eps,p.std,p.D,p.T,p.size))
    sims = pd.DataFrame(sims)
    sims = combine_simulations(sims,lsims)
    store_cache(sims,cache_name)

    return sims


def run_frame_v_mean_sim(grid,parallel=True,rerun=False):
    cache_name = f"hb_gaussian_fvm"
    lsims = load_cache(cache_name)

    # f_grid = []
    # f_grid = grid
    grid = filter_grid_field(grid,'B')
    f_grid = filter_complete_exps(grid,lsims)
    # rerun = True
    if rerun is False and len(f_grid) == 0: return lsims
    print(f"Simulating [{len(f_grid)}] Experimental Setups with Cache [{cache_name}]")

    if parallel:
        pParallel = ProgressParallel(True,len(f_grid),n_jobs=8)
        delayed_fxn = delayed(sim_frame_v_mean)
        sims = pParallel(delayed_fxn(p.eps,p.std,p.D,p.T,p.size)
                         for p in f_grid)
    else:
        sims = []
        for p in f_grid:
            sims.append(sim_frame_v_mean(p.eps,p.std,p.D,p.T,p.size))
    sims = pd.DataFrame(sims)
    sims = combine_simulations(sims,lsims)
    store_cache(sims,cache_name)

    return sims



def run_bootstrap_sim(grid,parallel=True,rerun=False):

    cache_name = f"hb_gaussian_bootstrap"
    # cache_name = f"proposed_{B}"
    # cache_name = f"proposed_tmp"
    lsims = load_cache(cache_name)

    f_grid = []
    # f_grid = grid
    # f_grid = filter_complete_exps(grid,lsims)
    # rerun = True
    if rerun is False and len(f_grid) == 0:
        print(f"Loaded all sims from cache [{cache_name}]")
        return lsims

    print(f"Simulating [{len(f_grid)}] Experimental Setups")
    
    if parallel:
        # pParallel = Parallel(n_jobs=8)
        pParallel = ProgressParallel(True,len(f_grid),n_jobs=8)
        delayed_fxn = delayed(sim_bootstrap)
        sims = pParallel(delayed_fxn(p.eps,p.std,p.D,p.T,p.B,p.size)
                         for p in f_grid)
    else:
        sims = []
        for p in tqdm(f_grid):
            sims.append(sim_bootstrap(p.eps,p.std,p.D,p.T,p.B,p.size))
    sims = pd.DataFrame(sims)
    sims = combine_simulations(sims,lsims)
    store_cache(sims,cache_name)

    return sims


def combine_simulations(sims,lsims):
    return pd.concat([sims,lsims],axis=0)

