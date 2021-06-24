
# -- python imports --
import numpy as np
import pandas as pd
import numpy.random as npr

# -- [local] imports --
from .cache import store_cache,load_cache,get_cache_name
from .parallel import ProgressParallel

def sim_proposed_test_data(pgrid,parallel=True):
    sims = load_cache("proposed")
    if not(sims is None): return sims

    def sim_proposed_single(D,pmis,ub,std,T,B,size):

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
        
        def sim_gaussian2(D,mu2,std,size):
            gaussian_std = np.sqrt(4* mu2 * std**2)/D
            x = npr.normal(loc=mu2,scale=gaussian_std,size=size)
    
            gamma_shape = D/2
            gamma_scale = 2*(std**2)/D
            y = npr.gamma(gamma_shape,scale=gamma_scale,size=size)
            return x+y

        def sim_v1(D,pmis,ub,std,T,B,size):
            samples_pix = np.zeros(size)
            samples_zero = np.zeros(size)
            mu2_samples = np.zeros(size)
            
            # -- 1.) simulate misalignment --
            nmis = int(T * pmis)
            if nmis == 0 and not np.isclose(pmis,0):
                raise ValueError("At least one should be misaligned.")
            mis = npr.uniform(0,ub,(nmis,D,size))
            pix = np.zeros((T,D,size))
            pix[:nmis] = mis

            # -- 2.) simulate \hat{MSE}(\bar{X}) --
            zeros = np.zeros(size)
            pix_mean = np.mean(pix,axis=0)
            for b in range(B):

                # -- subset and count unique --
                subset = npr.choice(T,(T,size)).astype(np.int)
                n_unique = count_unique_cols(subset)

                # -- subset mu2 --
                pix_smean = np.zeros((D,size))
                for s in range(size):
                    pix_smean[:,s] = np.mean(pix[subset[:,s],:,s],axis=0)
                mu2_b = np.mean((pix_smean - pix_mean)**2,axis=0)
                mu2_samples += mu2_b
                
                # -- subset std --
                std_s = std / n_unique
                std_T = std / T
                std_b = std_s + std_T

                # -- simulate from "pix"-misaligned distribution --
                sample_pix = sim_gaussian2(D,mu2_b,std_b,1)
                samples_pix += sample_pix

                # -- simulate from "aligned" distribution --
                sample_zero = sim_gaussian2(D,zeros,std_b,1)
                samples_zero += sample_zero
            samples_pix /= B
            samples_zero /= B
            mu2_samples /= B
            cond = samples_zero < samples_pix
            return cond,mu2_samples

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
    sims = store_cache(sims,"proposed")

    return sims

