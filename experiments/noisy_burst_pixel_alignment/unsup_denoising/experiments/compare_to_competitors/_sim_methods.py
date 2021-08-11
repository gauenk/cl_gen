import torch
import numba
from numba import cuda
from numba.typed import List
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_uniform_float32
from einops import rearrange,repeat

# -- python --
import numpy as np
from ._aligned_methods import get_align_method
    
def get_sim_method(cfg,method_name):
    aligned_fxn = get_align_method(cfg,method_name)
    def sim_fxn(burst,sim_type,db=None,gt_info=None):
        aligned,flow = None,None
        if sim_type == "a":
            aligned,flow = aligned_fxn(burst,db,gt_info)
            sims,masks = uniform_pix_sampling(aligned)
        elif sim_type == "b":
            aligned,flow = aligned_fxn(burst,db,gt_info)
            nframes = aligned.shape[0]
            ref = nframes//2
            aligned = torch.stack([aligned[:ref],aligned[ref+1:]])
            sims,masks = uniform_pix_sampling(aligned)
            sims[0] = burst[nframes//2]
        elif sim_type == "sup":
            sims,masks = get_gt_clean_sim(burst,gt_info)
        elif sim_type == "n2n":
            masks = None
            sims,masks = get_gt_noisy_sim(burst,gt_info)
        else:
            raise ValueError(f"Uknown sim type [{sim_type}]")
        return sims,masks,aligned,flow
    return sim_fxn

def uniform_pix_sampling(aligned,S=2):
    nframes,nimages,ncolor,h,w = aligned.shape
    device = aligned.device
    gpuid = int(str(device).split(":")[1])
    numba.cuda.select_device(gpuid)
    sims = torch.zeros((S,nimages,ncolor,h,w)).to(device)
    # rands = np.random.choice(nframes,(h,w))
    index_bursts_by_frames(sims,aligned)
    masks = torch.zeros((S,nimages,ncolor,h,w)).to(device)
    fill_masks(masks,aligned)
    return sims,masks

def fill_masks(masks,aligned):
    S = masks.shape[0]
    nframes,nimages,ncolor,H,W = aligned.shape
    threads_per_block = (32,32)
    blocks_H = H//threads_per_block[0] + (H%threads_per_block[0] != 0)
    blocks_W = W//threads_per_block[1] + (W%threads_per_block[1] != 0)
    blocks = (blocks_H,blocks_W)
    nthreads = int(np.product([blocks[i] * threads_per_block[i] for i in range(2)]))
    seed = int(torch.rand(1)*100)
    rng_states = create_xoroshiro128p_states(nthreads,seed=seed)
    fill_masks_numba[blocks,threads_per_block](rng_states,nimages,
                                               ncolor,H,W,S,
                                               nframes,aligned,masks)
    
@cuda.jit
def fill_masks_numba(rng_states,nimages,ncolor,H,W,S,nframes,aligned,masks):
    pass


def index_bursts_by_frames(sims,aligned):
    S = sims.shape[0]
    nframes,nimages,ncolor,H,W = aligned.shape
    threads_per_block = (32,32)
    blocks_H = H//threads_per_block[0] + (H%threads_per_block[0] != 0)
    blocks_W = W//threads_per_block[1] + (W%threads_per_block[1] != 0)
    blocks = (blocks_H,blocks_W)
    nthreads = int(np.product([blocks[i] * threads_per_block[i] for i in range(2)]))
    seed = int(torch.rand(1)*100)
    rng_states = create_xoroshiro128p_states(nthreads,seed=seed)
    uniform_pix_sample_by_frames_numba[blocks,threads_per_block](rng_states,nimages,
                                                                 ncolor,H,W,S,
                                                                 nframes,aligned,sims)

@cuda.jit
def uniform_pix_sample_by_frames_numba(rng_states,nimages,ncolor,
                                       H,W,S,nframes,aligned,sims):

    # -- get kernel indices --
    r_idx = cuda.grid(1)
    h_idx,w_idx = cuda.grid(2)
    subset = cuda.local.array(shape=51,dtype=numba.uint8) # MAX FRAMES = 51
    if h_idx < aligned.shape[0] and w_idx < aligned.shape[1]:

        # -- assign rand pix to sims --
        for s in range(S):
            # -- sim random frame --
            rand_float = nframes*xoroshiro128p_uniform_float32(rng_states, r_idx)
            rand_t = 0
            while rand_t < rand_float:
                rand_t += 1
            rand_t -= 1
            for i in range(nimages):
                for c in range(ncolor):                
                    sims[s][i][c][h_idx][w_idx] = aligned[rand_t][i][c][h_idx][w_idx]

def get_gt_clean_sim(burst,gt_info):
    dyn_clean = gt_info['dyn_clean']
    T,B,C,H,W = dyn_clean.shape
    ref_t = T//2
    clean = torch.stack([burst[ref_t],dyn_clean[ref_t]])
    masks = None
    return clean,masks

def get_gt_noisy_sim(burst,gt_info):
    static = gt_info['static_noisy']
    T,B,C,H,W = static.shape
    ref_t = T//2
    noisy = torch.stack([burst[ref_t],static[ref_t]])
    masks = None
    return noisy,masks
    
