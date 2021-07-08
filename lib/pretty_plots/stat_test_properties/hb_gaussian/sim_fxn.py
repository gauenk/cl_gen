# -- python imports --
import numpy as np
import numpy.random as npr
from einops import repeat
from easydict import EasyDict as edict

# -- project imports --
from pyutils import numba_unique,numba_subset_mean_along_axis,numba_compute_muB,numba_compute_deltas_bs

def sim_bootstrap(eps,std,D,T,B,size):

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
    
    def sim_normal2(D,mu2,std):
        normal_std = 2 * std * np.sqrt( mu2 ) / D
        x = npr.normal(loc=mu2,scale=normal_std)

        gamma_shape = D/2
        gamma_scale = 2*(std**2)/D
        y = npr.gamma(gamma_shape,scale=gamma_scale)
        return x+y

    def sim_clean_frames(eps,T,D,size):
        return npr.normal(loc=0,scale=eps,size=(T,D,size))

    def sim_noisy_frames(clean,std):
        return npr.normal(loc=clean,scale=std)
        
    def sim_v1(eps,std,D,T,B,S):

        # -- setup refs --
        smean = np.zeros((D,B,S))
        clean_deltas = np.zeros((D,B,S))
        noisy_deltas = np.zeros((D,B,S))
        
        # -- (a) simulate clean frames --
        clean = sim_clean_frames(eps,T,D,S)

        # -- (b) simulate noisy frames --
        noisy = sim_noisy_frames(clean,std)

        # -- (c) create subsets --
        subsets = npr.choice(T,(T,D,B,S)).astype(np.int)

        # -- (d) compute subset deltas --
        mean = np.mean(noisy,axis=0)
        numba_compute_deltas_bs(noisy_deltas,noisy,smean,mean,subsets)
        mean = np.mean(clean,axis=0)
        numba_compute_deltas_bs(clean_deltas,clean,smean,mean,subsets)

        # deltas.shape = (D,B,S)
        # pix.shape = (T,D,S)
        # pix_smean.shape = (D,B,S)
        # pix_mean.shape = (D,S)
        # subsets = (T,D,B,S)

        # -- (e) mean over Bootstrap --
        noisy_deltas = np.mean(noisy_deltas,axis=1)
        clean_deltas = np.mean(clean_deltas,axis=1)

        return noisy_deltas,clean_deltas

    noisy,clean = sim_v1(eps,std,D,T,B,size)
    sim = edict()
    sim.noisy = noisy
    sim.clean = clean

    # -- include parameters --
    sim.eps = eps
    sim.std = std
    sim.D = D
    sim.T = T
    sim.B = B
    sim = dict(sim)

    return sim

def sim_frame_v_frame(eps,std,D,T,S):

    def sim_clean_frames(eps,T,D,S):
        return npr.normal(loc=0,scale=eps,size=(T,D,S))

    def sim_noisy_frames(clean,std):
        return npr.normal(loc=clean,scale=std)

    def sim_v1(eps,std,D,T,S):

        # -- (a) simulate clean frames --
        clean = sim_clean_frames(eps,T,D,S)

        # -- (b) simulate noisy frames --
        noisy = sim_noisy_frames(clean,std)

        # -- (c) frame v frame difference --
        ref_T = repeat(noisy[[T//2]],'1 D S -> Tm1 D S',Tm1=T-1)
        noisy_nomid = np.r_[noisy[:T//2],noisy[T//2+1:]]
        noisy_deltas = np.mean( (noisy_nomid - ref_T)**2, axis=0)

        ref_T = repeat(clean[[T//2]],'1 D S -> Tm1 D S',Tm1=T-1)
        clean_nomid = np.r_[clean[:T//2],clean[T//2+1:]]
        clean_deltas = np.mean( (clean_nomid - ref_T)**2, axis=0)

        return noisy_deltas,clean_deltas

    noisy,clean = sim_v1(eps,std,D,T,S)
    sim = edict()
    sim.noisy = noisy
    sim.clean = clean

    # -- include parameters --
    sim.eps = eps
    sim.std = std
    sim.D = D
    sim.T = T
    sim = dict(sim)

    return sim

def sim_frame_v_mean(eps,std,D,T,S):

    def sim_clean_frames(eps,T,D,S):
        return npr.normal(loc=0,scale=eps,size=(T,D,S))

    def sim_noisy_frames(clean,std):
        return npr.normal(loc=clean,scale=std)

    def sim_v1(eps,std,D,T,S):

        # -- (a) simulate clean frames --
        clean = sim_clean_frames(eps,T,D,S)

        # -- (b) simulate noisy frames --
        noisy = sim_noisy_frames(clean,std)

        # -- (c) frame v frame difference --
        ave = np.mean(noisy,axis=0)
        ave = repeat(ave,'D S -> T D S',T=T)
        noisy_deltas = np.mean( (noisy - ave)**2, axis=0)

        ave = np.mean(clean,axis=0)
        ave = repeat(ave,'D S -> T D S',T=T)
        clean_deltas = np.mean( (clean - ave)**2, axis=0)

        return noisy_deltas,clean_deltas

    noisy,clean = sim_v1(eps,std,D,T,S)
    sim = edict()
    sim.noisy = noisy
    sim.clean = clean

    # -- include parameters --
    sim.eps = eps
    sim.std = std
    sim.D = D
    sim.T = T
    sim = dict(sim)

    return sim


