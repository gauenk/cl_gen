
# -- python imports --
import math
import numpy as np
import pandas as pd
import numpy.random as npr
from joblib import delayed
from easydict import EasyDict as edict

# -- local imports --
from pretty_plots.stat_test_properties.cache import store_cache,load_cache,get_cache_name
from pretty_plots.stat_test_properties.parallel import ProgressParallel

def get_standard_sims(pgrid,parallel=True):

    sims = load_cache("standard")
    rerun = False
    if not(rerun) and not(sims is None): return sims

    def sim_standard_single(D,mu2,std,T,reps):

        def sim_v1(D,mu2,std,T,reps):

            gaussian_std = math.sqrt( 4 * mu2 * std**2 )/D
            x = npr.normal(loc=mu2,scale=gaussian_std,size=(T,reps))
    
            gamma_shape = D/2
            gamma_scale = 2*(std**2)/D
            y = npr.gamma(gamma_shape,scale=gamma_scale,size=(T,reps))
            z = npr.gamma(gamma_shape,scale=gamma_scale,size=(T,reps))

            left = np.mean(z,axis=0)
            right = np.mean(y + x,axis=0)
            cond = left < right
            return cond

        def sim_v2(D,mu2,std,T,reps):
            D = int(D)

            left = npr.normal(loc=0,scale=std,size=(D,T,reps))**2
            left = np.mean(np.mean(left,axis=0),axis=0)

            right = npr.normal(loc=math.sqrt(mu2),scale=std,size=(D,T,reps))**2
            right = np.mean(np.mean(right,axis=0),axis=0)

            cond = left < right
            return cond
    
        cond = sim_v1(D,mu2,2*std,T,reps)
        sim = edict()
        sim.est_mean = np.mean(cond)
        sim.est_std = np.std(cond)
    
        # -- include parameters --
        sim.mu2 = mu2
        sim.std = std
        sim.D = D
        sim.T = T
        sim = dict(sim)

        return sim
    
    if parallel:
        # pParallel = Parallel(n_jobs=8)
        pParallel = ProgressParallel(True,len(pgrid),n_jobs=8)
        sims = pParallel(delayed(sim_standard_single)(p.D,p.mu2,p.std,p.T,p.size)
                                   for p in pgrid)
    else:
        sims = []
        for p in pgrid:
            sims.append(sim_standard_single(p.D,p.mu2,p.std,p.T,p.size))
    sims = pd.DataFrame(sims)
    store_cache(sims,"standard")
    return sims
