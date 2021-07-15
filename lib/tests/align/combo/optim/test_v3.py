

# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
import settings
import align.nnf as nnf
import align.combo as combo
import align.combo.optim as optim
from align.xforms import align_burst_from_flow_padded_patches,flow_to_blocks,align_burst_from_blocks_padded_patches
from pyutils import tile_patches
from patch_search import get_score_function
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

from align._utils import torch_to_numpy

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def config():
    cfg = edict()

    # -- exp settings --
    cfg.nframes = 3

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.batch_size = 2
    # cfg.dataset.load_residual = False
    # cfg.dataset.triplet_loader = True
    # cfg.dataset.bw = False

    cfg.noise_params = edict({'g':{'std':50},'ntype':'g'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'global'
    cfg.dynamic_info.frame_size = 16
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 1
    cfg.random_seed = 123

    # -- combo config --
    cfg.nblocks = 5
    cfg.patchsize = 3
    cfg.score_fxn_name = "bootstrapping"
    
    return cfg

def check_parameters(nblocks,patchsize):
    even_blocks = nblocks % 2 == 0
    even_ps = patchsize % 2 == 0
    if even_blocks or even_ps:
        print("Even number of blocks or patchsizes. We recommend odd so a center exists.")

def test_nnf():

    # -- get config --
    cfg = config()

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load dataset --
    data,loaders = load_image_dataset(cfg)
    image_iter = iter(loaders.tr)    

    # -- get score function --
    score_fxn = get_score_function(cfg.score_fxn_name)

    # -- some constants --
    NUM_BATCHES = 2
    nframes,nblocks = cfg.nframes,cfg.nblocks 
    patchsize = cfg.patchsize
    check_parameters(nblocks,patchsize)

    # -- create evaluator
    iterations,K = 2,2
    subsizes = [3,2]
    evaluator = combo.eval_scores.EvalBlockScores(score_fxn,patchsize,100,None)

    # -- iterate over images --
    for image_bindex in range(NUM_BATCHES):

        # -- sample & unpack batch --
        sample = next(image_iter)
        sample_to_cuda(sample)
        dyn_noisy = sample['noisy'] # dynamics and noise
        dyn_clean = sample['burst'] # dynamics and no noise
        static_noisy = sample['snoisy'] # no dynamics and noise
        static_clean = sample['sburst'] # no dynamics and no noise
        flow = sample['flow']
        
        # -- shape info --
        T,B,C,H,W = dyn_noisy.shape

        # -- compute nearest neighbor fields --
        ref_t = nframes//2
        nnf_vals,nnf_pix = nnf.compute_burst_nnf(dyn_clean,ref_t,patchsize)
        print("[nnf_pix.shape]: ",nnf_pix.shape)
        nnf_pix_best = rearrange(nnf_pix[...,0,:],'b t h w two -> b (h w) t two')
        print("[nnf_pix.shape]: ",nnf_pix.shape)
        # nnf_flow_best = flow_to_blocks(nnf_pix_best,nblocks)
        
        # -- prepare patches --
        pad = 2*(nblocks//2)
        h,w = patchsize+pad,patchsize+pad
        patches = tile_patches(dyn_clean,patchsize+pad).pix
        patches = rearrange(patches,'b t s (c h w) -> b s t c h w',h=h,w=w)
        masks = torch.ones_like(patches).type(torch.long)

        # -- run optimization --
        # est_flow = optim.v3.run_image_batch(patches,masks,evaluator,
        #                                     nblocks,iterations,
        #                                     subsizes,K)
        # est_flow = rearrange(est_flow,'t i (h w) two -> i t h w 1 two',h=H)

        

        # -- print for fun --
        print("[patches.shape]: ",patches.shape)
        print("[nnf_pix.shape]: ",nnf_pix.shape)
        print("[nnf_pix_best.shape]: ",nnf_pix_best.shape)
        # print("[nnf_flow.shape]: ",nnf_flow.shape)
        # print("[est_flow.shape]: ",est_flow.shape)
        print("[flow.shape]: ",flow.shape)

        # -- compare nnf v.s. est --
        # est_of = compute_epe_error(est_flow,flow)
        # nnf_of = compute_epe_error(nnf_flow,flow)
        # est_nnf = compute_epe_error(nnf_flow,flow)
        # print("EPE Errors")
        # print(est_of)
        # print(nnf_of)
        # print(est_nnf)

        # -- compare psnr of nnf v.s. est aligned images --
        # nnf_flow_best = nnf_flow[...,0,:]
        # print("nnf_flow_best.shape",nnf_flow_best.shape)
        # nnf_flow_best = rearrange(nnf_flow_best,'b t h w two -> b (h w) t two')

        # print("[stats of nnf_flow_best]: ",nnf_flow_best.min(),nnf_flow_best.max())
        # print("[(post) nnf_flow_best.shape]: ",nnf_flow_best.shape)

        print("[stats of nnf_pix_best]: ",nnf_pix_best.min(),nnf_pix_best.max())
        print("[(post) nnf_pix_best.shape]: ",nnf_pix_best.shape)

        patches = torch_to_numpy(patches).astype(np.float)

        print("[(pre) patches.shape]: ",patches.shape)
        # aligned = align_burst_from_blocks_padded_patches(patches,nnf_pix_best,
        #                                                  nblocks,patchsize)
        print("[aligned.shape]: ",aligned.shape)
        
