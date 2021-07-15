

# -- python imports --
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF


# -- project imports --
import settings
from align import compute_epe,compute_aligned_psnr
import align.nnf as nnf
import align.combo as combo
import align.combo.optim as optim
from align.xforms import align_from_pix,flow_to_pix,create_isize,pix_to_flow,align_from_flow
from pyutils import tile_patches,save_image
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
    cfg.batch_size = 5
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
    iterations,K = 5,3
    subsizes = [1,1,1]
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
        flow_gt = sample['flow']

        # -- shape info --
        T,B,C,H,W = dyn_noisy.shape
        ref_t = nframes//2
        npix = H*W

        # -- groundtruth flow --
        flow_gt = repeat(flow_gt,'i tm1 two -> i p tm1 two',p=npix)
        print("sample['flow']: ",flow_gt.shape)
        
        # -- compute nearest neighbor fields --
        shape_str = 't b h w two -> b (h w) t two'
        nnf_vals,nnf_pix = nnf.compute_burst_nnf(dyn_clean,ref_t,patchsize)
        nnf_pix_best = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        nnf_pix_best = torch.LongTensor(nnf_pix_best)
        flow_nnf = pix_to_flow(nnf_pix_best)
        aligned_nnf = align_from_pix(dyn_clean,nnf_pix_best,patchsize)

        # -- prepare patches --
        flow_est = optim.v3.run_image_burst(dyn_clean,patchsize,evaluator,
                                            nblocks,iterations,subsizes,K)
        isize = edict({'h':H,'w':W})
        aligned_est = align_from_flow(dyn_clean,flow_est,patchsize,isize=isize)

        # -- compare nnf v.s. est --
        est_of = compute_epe(flow_est,flow_gt)
        nnf_of = compute_epe(flow_nnf,flow_gt)
        est_nnf = compute_epe(flow_est,flow_nnf)
        print("EPE Errors")
        print(est_of)
        print(nnf_of)
        print(est_nnf)

        # -- eval --

        pad = 2*patchsize
        isize = edict({'h':H-pad,'w':W-pad})
        psnr_nnf = compute_aligned_psnr(aligned_nnf,static_clean,isize)
        psnr_est = compute_aligned_psnr(aligned_est,static_clean,isize)
        print("PSNR Values")
        print(psnr_nnf)
        print(psnr_est)

        # cc_aligned_nnf = tvF.center_crop(aligned_nnf,(H-pad,W-pad))
        # cc_static_clean = tvF.center_crop(static_clean,(H-pad,W-pad))

        # print("aligned_nnf.shape ",aligned_nnf.shape)
        # delta = torch.sum(torch.abs(cc_static_clean.cpu() - cc_aligned_nnf.cpu())).item()
        # print("static_clean.shape ",static_clean.shape)
        # print("delta ", delta)

        # -- compare psnr of nnf v.s. est aligned images --
        # nnf_flow_best = nnf_flow[...,0,:]
        # print("nnf_flow_best.shape",nnf_flow_best.shape)
        # nnf_flow_best = rearrange(nnf_flow_best,'b t h w two -> b (h w) t two')

        # print("[stats of nnf_flow_best]: ",nnf_flow_best.min(),nnf_flow_best.max())
        # print("[(post) nnf_flow_best.shape]: ",nnf_flow_best.shape)

        # print("[stats of nnf_pix_best]: ",nnf_pix_best.min(),nnf_pix_best.max())
        # print("[(post) nnf_pix_best.shape]: ",nnf_pix_best.shape)

        # patches = torch_to_numpy(patches).astype(np.float)

        # print("[(pre) patches.shape]: ",patches.shape)
        # # aligned = align_burst_from_blocks_padded_patches(patches,nnf_pix_best,
        # #                                                  nblocks,patchsize)
        # print("[aligned.shape]: ",aligned.shape)
        
