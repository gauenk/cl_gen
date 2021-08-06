# -- python imports --
import copy
import numpy as np
import numpy.random as npr
from einops import rearrange,repeat
from joblib import delayed

# -- pytorch imports --
import torch

# -- imports --
from pyutils import tile_patches,save_image
from align._parallel import ProgressParallel
from align._utils import BatchIter

# ------------------------------------
#
#       Main Interface Function
#
# ------------------------------------

def run_burst(fxn,burst,patchsize,evaluator,
                    nblocks,iterations,
                    subsizes,K):
    r"""
    Run optimization on a burst of images
    a.) tile image
    b.) run over patch batches
    """

    pad = 2*(nblocks//2)
    #pad = nblocks
    h,w = patchsize+pad,patchsize+pad
    patches = tile_patches(burst,patchsize+pad).pix
    patches = rearrange(patches,'b t s (h w c) -> b s t c h w',h=h,w=w)
    # patches = rearrange(patches,'b t s (w h c) -> b s t c h w',h=h,w=w)
    masks = torch.ones_like(patches).type(torch.long)
    # save_image(patches[0,0],"patches_0.png",(-0,5,0.5))
    # save_image(patches[0,32*15],"patches_31x15.png",(-0,5,0.5))
    # save_image(patches[0,32*16],"patches_32x16.png",(-0,5,0.5))
    # exit()
    torch.cuda.empty_cache()
    # print("patches.device ",patches.device)

    return run_image_batch(fxn,patches,masks,evaluator,
                           nblocks,iterations,
                           subsizes,K)

# ------------------------------------
#
#   Run Optim over Batches of Images
#
# ------------------------------------

def run_image_batch(fxn,patches,masks,evaluator,
                    nblocks,iterations,
                    subsizes,K):
    
    r"""
    Split computation in parallel across image batches

    choose parallel or serial
    """

    PARALLEL = False
    if PARALLEL:
        return run_image_batch_parallel(fxn,patches,masks,evaluator,
                                        nblocks,iterations,
                                        subsizes,K)
    else:
        return run_image_batch_serial(fxn,patches,masks,evaluator,
                                      nblocks,iterations,
                                      subsizes,K)

def run_image_batch_parallel(fxn,patches,masks,evaluator,
                             nblocks,iterations,
                             subsizes,K):
    r"""
    Run 
    """

    blocks = []
    nimages = patches.shape[0]
    pParallel = ProgressParallel(False,len(patches),n_jobs=8)
    delayed_fxn = delayed(run_pixel_batch)
    blocks = pParallel(delayed_fxn(fxn,patches[[i]],masks[[i]],evaluator,
                                   nblocks,iterations,subsizes,K)
                       for i in range(nimages))
    blocks = torch.cat(blocks,dim=0) # nimages, npix, nframes-1, 2
    return blocks

def run_image_batch_serial(fxn,patches,masks,evaluator,
                           nblocks,iterations,
                           subsizes,K):
    flows = []
    nimages = patches.shape[0]
    for b in range(nimages):
        flow_b = run_pixel_batch(fxn,patches[[b]],masks[[b]],evaluator,
                                 nblocks,iterations,subsizes,K)        
        flows.append(flow_b)
    flows = torch.cat(flows) # nimages, npix, nframes-1, 2
    return flows

# ------------------------------------
#
#   Run Optim over Batches of Pixels
#
# ------------------------------------


def run_pixel_batch(fxn,patches,masks,evaluator,
                    nblocks,iterations,
                    subsizes,K):
    
    r"""
    Split computation in parallel across image batches

    choose parallel or serial
    """
    evaluator = copy.deepcopy(evaluator)
    PARALLEL = True
    # if evaluator.score_fxn_name == "ave":
    #     PARALLEL = True
    # else:
    #     PARALLEL = False
    if PARALLEL:
        return run_pixel_batch_parallel(fxn,patches,masks,evaluator,
                                        nblocks,iterations,
                                        subsizes,K)
    else:
        return run_pixel_batch_serial(fxn,patches,masks,evaluator,
                                      nblocks,iterations,
                                      subsizes,K)

def run_pixel_batch_parallel(fxn,patches,masks,evaluator,
                             nblocks,iterations,
                             subsizes,K):
    
    nimages,npix,nframes,c,h,w = patches.shape
    nimages,npix,nframes,c,h,w = masks.shape

    # -- alg. comp. efficiency --
    if evaluator.score_fxn_name == "ave":
        if nframes > 10:
            PIX_BATCHSIZE = 64
            N_JOBS = 6
        else:
            PIX_BATCHSIZE = 128
            N_JOBS = 4
    elif evaluator.score_fxn_name == "bs":
        if nframes < 10:
            if h <= 7:
                PIX_BATCHSIZE = 128
                N_JOBS = 8
            else:
                PIX_BATCHSIZE = 128
                N_JOBS = 6
        elif nframes == 20:
            PIX_BATCHSIZE = 128
            N_JOBS = 4
        else:
            PIX_BATCHSIZE = 64
            N_JOBS = 4
    else:
        # print("eval",evaluator.score_fxn_name)
        PIX_BATCHSIZE = 128
        N_JOBS = 4
    
    # print(h,PIX_BATCHSIZE,N_JOBS)
    piter = BatchIter(npix,PIX_BATCHSIZE)

    flows = []
    pParallel = ProgressParallel(False,len(piter),n_jobs=N_JOBS)
    delayed_fxn = delayed(fxn)
    flows = pParallel(delayed_fxn(patches[:,pbatch],masks[:,pbatch],evaluator,
                                  nblocks,iterations,subsizes,K,p)
                      for p,pbatch in enumerate(piter))
    flows = torch.cat(flows,dim=1) # nimages, npix, nframes-1, 2
    return flows

def run_pixel_batch_serial(fxn,patches,masks,evaluator,
                           nblocks,iterations,
                           subsizes,K):
    
    PIX_BATCHSIZE = 16
    nimages,npix,nframes,c,h,w = patches.shape
    nimages,npix,nframes,c,h,w = masks.shape

    piter = BatchIter(npix,PIX_BATCHSIZE)
    flows = []
    for p,pbatch in enumerate(piter):
        flow_p = fxn(patches[:,pbatch],masks[:,pbatch],evaluator,
                     nblocks,iterations,subsizes,K,p)
        flows.append(flow_p)
    flows = torch.cat(flows,dim=1)
    return flows


