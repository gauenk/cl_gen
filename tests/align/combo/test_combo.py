

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
from pyutils import tile_patches,save_image,torch_to_numpy
from patch_search import get_score_function
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def config():
    cfg = edict()

    # -- exp settings --
    cfg.nframes = 10

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
    iterations,K = 10,2
    subsizes = [2,2,2,2,2]
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
        isize = edict({'h':H,'w':W})
        ref_t = nframes//2
        npix = H*W

        # -- groundtruth flow --
        flow_gt = repeat(flow_gt,'i tm1 two -> i p tm1 two',p=npix)
        print("sample['flow']: ",flow_gt.shape)
        aligned_of = align_from_flow(dyn_clean,flow_gt,patchsize,isize=isize)
        
        # -- compute nearest neighbor fields --
        shape_str = 't b h w two -> b (h w) t two'
        nnf_vals,nnf_pix = nnf.compute_burst_nnf(dyn_clean,ref_t,patchsize)
        nnf_pix_best = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        nnf_pix_best = torch.LongTensor(nnf_pix_best)
        flow_nnf = pix_to_flow(nnf_pix_best)
        aligned_nnf = align_from_pix(dyn_clean,nnf_pix_best,patchsize)

        # -- compute proposed search of nnf --
        flow_split = optim.v1.run_image_burst(dyn_clean,patchsize,evaluator,
                                              nblocks,iterations,subsizes,K)
        isize = edict({'h':H,'w':W})
        aligned_split = align_from_flow(dyn_clean,flow_split,patchsize,isize=isize)

        # -- compute proposed search of nnf --
        flow_est = optim.v3.run_image_burst(dyn_clean,patchsize,evaluator,
                                            nblocks,iterations,subsizes,K)
        aligned_est = align_from_flow(dyn_clean,flow_est,patchsize,isize=isize)

        # -- banner --
        print("-"*25 + " Results " + "-"*25)

        # -- compare gt v.s. nnf computations --
        nnf_of = compute_epe(flow_nnf,flow_gt)
        split_of = compute_epe(flow_split,flow_gt)
        est_of = compute_epe(flow_est,flow_gt)

        split_nnf = compute_epe(flow_split,flow_nnf)
        est_nnf = compute_epe(flow_est,flow_nnf)

        print("-"*50)
        print("EPE Errors")
        print("-"*50)
        print("NNF v.s. Optical Flow.")
        print(nnf_of)
        print("Split v.s. Optical Flow.")
        print(split_of)
        print("Proposed v.s. Optical Flow.")
        print(est_of)
        print("Split v.s. NNF")
        print(split_nnf)
        print("Proposed v.s. NNF")
        print(est_nnf)


        # -- psnr eval --
        pad = 2*patchsize
        isize = edict({'h':H-pad,'w':W-pad})
        psnr_of = compute_aligned_psnr(aligned_of,static_clean,isize)
        psnr_nnf = compute_aligned_psnr(aligned_nnf,static_clean,isize)
        psnr_split = compute_aligned_psnr(aligned_split,static_clean,isize)
        psnr_est = compute_aligned_psnr(aligned_est,static_clean,isize)
        print("-"*50)
        print("PSNR Values")
        print("-"*50)
        print("Optical Flow [groundtruth v1]")
        print(psnr_of)
        print("NNF [groundtruth v2]")
        print(psnr_nnf)
        print("Split [old method]")
        print(psnr_split)
        print("Proposed [new method]")
        print(psnr_est)

