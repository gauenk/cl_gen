
# -- python imports --
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy.random as npr
from joblib import delayed
from easydict import EasyDict as edict

# -- project imports --
from pyutils import numba_unique,numba_subset_mean_along_axis,numba_compute_muB

# -- [local] project imports --
from pretty_plots.stat_test_properties.cache import store_cache,load_cache,get_cache_name
from pretty_plots.stat_test_properties.parallel import ProgressParallel

def numba_mat_count_uniques_bs(mat,n_uniques):
    B,T,S = mat.shape
    for b in range(B):
        for s in range(S):
            q = np.zeros(256, dtype=int)
            n_unique = numba_unique(mat[b,:,s],q)
            n_uniques[b,s] = n_unique

def filter_pandas_by_dict(df,pydict):
    compare = df[list(pydict)] == pd.Series(pydict)
    loc = compare.all(1)
    return df.loc[loc]

def get_proposed_sims(pgrid,parallel=True,rerun=False):
    B = pgrid[0].B

    cache_name = f"proposed"
    # cache_name = f"proposed_{B}"
    # cache_name = f"proposed_tmp"
    lsims = load_cache(cache_name)
    print(lsims['T'].unique())
    f_pgrid = filter_complete_exps(pgrid,lsims)
    
    # rerun = True
    if rerun is False and len(f_pgrid) == 0: return lsims
    print(f"Simulating [{len(f_pgrid)}] Experimental Setups")

    def sim_proposed_single(D,pmis,ub,std,T,B,size):

        def numba_subset_mean(smean,pix,subset):
            return numba_subset_mean_along_axis(smean,pix,subset)

        def numba_mat_count_uniques_bs(mat,n_uniques):
            B,T,S = mat.shape
            for b in range(B):
                for s in range(S):
                    q = np.zeros(256, dtype=int)
                    n_unique = numba_unique(mat[b,:,s],q)
                    n_uniques[b,s] = n_unique

        def numba_mat_count_uniques(mat,ncols):
            n_uniques = []
            for c in range(ncols):
                q = np.zeros(256, dtype=int)
                n_unique = numba_unique(mat[:,c],q)
                n_uniques.append(n_unique)
            return n_uniques
        
        def count_unique_cols(mat):
            ncols = mat.shape[1]
            n_uniques = numba_mat_count_uniques(mat,ncols)
            return np.array(n_uniques)
        
        def sim_gaussian2(D,mu2,std):
            gaussian_std = 2 * std * np.sqrt( mu2 ) / D
            x = npr.normal(loc=mu2,scale=gaussian_std)
    
            gamma_shape = D/2
            gamma_scale = 2*(std**2)/D
            y = npr.gamma(gamma_shape,scale=gamma_scale)
            return x+y

        def sim_v1(D,pmis,ub,std,T,B,size):
            
            # -- 1.) simulate misalignment --
            nmis = int(T * (pmis/100.))
            if nmis == 0 and not np.isclose(pmis,0):
                raise ValueError("At least one should be misaligned.")
            mis = npr.uniform(0,ub,(nmis,D,size))
            pix = np.zeros((T,D,size))
            pix[:nmis] = mis

            # -- setup refs --
            zerosB = np.zeros((B,size))
            zeros = np.zeros(size)
            pix_mean = np.mean(pix,axis=0)
            pix_smean = np.zeros((D,size))
            muB = np.zeros((B,size))
            n_uniques_B = np.zeros((B,size))

            #
            # -- 2.) simulate \hat{MSE}(\bar{X}) --
            #

            # -- (a) create subsets for each B and size --
            subset_B = npr.choice(T,(B,T,size)).astype(np.int)

            # -- (b) count unique for each trial along dim T --
            numba_mat_count_uniques_bs(subset_B,n_uniques_B)

            # -- (c) use # of unique to compuute std of sample "b" --
            std_B = std / n_uniques_B + std / T

            # -- (d) simulate "aligned" --
            samples_zero = np.mean(sim_gaussian2(D,zerosB,std_B),axis=0)

            # -- (e) simulate "misaligned" --
            numba_compute_muB(muB,pix,pix_smean,pix_mean,subset_B)
            samples_pix = np.mean(sim_gaussian2(D,muB,std_B),axis=0)

            cond = samples_zero < samples_pix
            return cond,muB

        # print("params: ",D,pmis,ub,std,T,B,size)
        cond,mu2_samples = sim_v1(D,pmis,ub,std,T,B,size)
        sim = edict()
        sim.est_mean = np.mean(cond)
        sim.est_std = np.std(cond)
        sim.est_mu2_mean = np.mean(mu2_samples)
        sim.est_mu2_std = np.std(mu2_samples)
    
        # -- include parameters --
        sim.pmis = pmis
        sim.ub = ub
        sim.std = std
        sim.D = D
        sim.T = T
        sim.B = B
        sim = dict(sim)

        return sim

    if parallel:
        # pParallel = Parallel(n_jobs=8)
        pParallel = ProgressParallel(True,len(f_pgrid),n_jobs=8)
        delayed_fxn = delayed(sim_proposed_single)
        sims = pParallel(delayed_fxn(p.D,p.pmis,p.ub,p.std,p.T,p.B,p.size)
                         for p in f_pgrid)
    else:
        sims = []
        for p in f_pgrid:
            sims.append(sim_proposed_single(p.D,p.pmis,p.ub,p.std,p.T,p.B,p.size))
    sims = pd.DataFrame(sims)
    sims = combine_simulations(sims,lsims)
    store_cache(sims,cache_name)

    return sims


def combine_simulations(sims,lsims):
    return pd.concat([sims,lsims],axis=0)
