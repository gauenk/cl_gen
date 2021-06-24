
# -- python imports --
import numpy as np
import pandas as pd
import numpy.random as npr
from joblib import delayed
from easydict import EasyDict as edict

# -- project imports --
from pyutils import numba_unique,numba_subset_mean_along_axis,numba_compute_muB

# -- local imports --
from .cache import store_cache,load_cache,get_cache_name
from .parallel import ProgressParallel


def sim_proposed_test_data(pgrid,parallel=True):
    sims = load_cache("proposed")
    if not(sims is None): return sims

    def sim_proposed_single(D,pmis,ub,std,T,B,size):

        def numba_subset_mean(smean,pix,subset):
            return numba_subset_mean_along_axis(smean,pix,subset)

        def numba_mat_count_uniques_bs(mat):
            B,T,S = mat.shape
            n_uniques = np.zeros((B,S))
            for b in range(B):
                for s in range(S):
                    q = np.zeros(256, dtype=int)
                    n_unique = numba_unique(mat[b,:,s],q)
                    n_uniques[b,s] = n_unique
            return n_uniques

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
            gaussian_std = np.sqrt(4* mu2 * std**2)/D
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

            #
            # -- 2.) simulate \hat{MSE}(\bar{X}) --
            #

            # -- (a) create subsets for each B and size --
            subset_B = npr.choice(T,(B,T,size)).astype(np.int)

            # -- (b) count unique for each trial along dim T --
            n_unique_B = numba_mat_count_uniques_bs(subset_B)

            # -- (c) use # of unique to compuute std of sample "b" --
            std_B = std / n_unique_B + std / T

            # -- (d) simulate "aligned" --
            samples_zero = np.mean(sim_gaussian2(D,zerosB,std_B),axis=0)

            # -- (e) simulate random numbers for "misaligned" --
            muB = np.zeros((B,size))
            numba_compute_muB(muB,pix_smean,pix,subset_B)
            
            samples_pix = sim_gaussian2(D,muB,std_B)
            numba_compute_muB(muB,pix_smean,pix,subset_B)

            def numba_compute_muB(muB,pix_smean,pix,subset_B):
                B = subset_B.shape[0]
                for b in range(B):
                    numba_subset_mean(pix_smean,pix,subset_B[b])
                    muB[b] = np.mean((pix_smean - pix_mean)**2,axis=0)

            for b in range(B):

                # -- subset and count unique --
                # subset = npr.choice(T,(T,size)).astype(np.int)
                # n_unique = count_unique_cols(subset)
                subset = subset_B[b]
                n_unique = n_unique_B[b]

                # -- subset mu2 --
                numba_subset_mean(pix_smean,pix,subset)
                # numba_subset_mean_along_axis_joblib(pix_smean,pix,subset)
                # for s in range(size):
                #     pix_smean[:,s] = np.mean(pix[subset[:,s],:,s],axis=0)
                mu2_b = np.mean((pix_smean - pix_mean)**2,axis=0)
                mu2_samples += mu2_b
                
                # -- subset std --
                # std_s = std / n_unique
                # std_T = std / T
                # std_b = std_s + std_T
                std_b = std_B[b]

                # -- simulate from "pix"-misaligned distribution --
                sample_pix = sim_gaussian2(D,mu2_b,std_b)
                samples_pix += sample_pix

                # -- simulate from "aligned" distribution --
                # sample_zero = sim_gaussian2(D,zeros,std_b)
                # samples_zero += sample_zero
                exit()
            cond = samples_zero < samples_pix
            return cond,muB

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
        pParallel = ProgressParallel(True,len(pgrid),n_jobs=8)
        delayed_fxn = delayed(sim_proposed_single)
        sims = pParallel(delayed_fxn(p.D,p.pmis,p.ub,p.std,p.T,p.B,p.size) for p in pgrid)
    else:
        sims = []
        for p in pgrid:
            sims.append(sim_proposed_single(p.D,p.pmis,p.ub,p.std,p.T,p.B,p.size))
    sims = pd.DataFrame(sims)
    store_cache(sims,"proposed")

    return sims

