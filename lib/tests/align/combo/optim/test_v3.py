

# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
import settings
import align.nnf as nnf
import align.combo as combo
import align.combo.optim as optim
from pyutils import tile_patches
from patch_search import get_score_function
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

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
    cfg.batch_size = 10
    # cfg.dataset.load_residual = False
    # cfg.dataset.triplet_loader = True
    # cfg.dataset.bw = False

    cfg.noise_params = edict({'g':{'std':50},'ntype':'g'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'global'
    cfg.dynamic_info.frame_size = 64
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 1
    cfg.random_seed = 123

    # -- combo config --
    cfg.nblocks = 5
    cfg.patchsize = 3
    cfg.score_fxn_name = "bootstrapping"
    
    return cfg

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

    # -- create evaluator
    iterations,K = 3,3
    subsizes = [3,2,2,2,2,2]
    evaluator = combo.eval_scores.EvalBlockScores(score_fxn,100,None)

    # -- some constants --
    NUM_BATCHES = 2
    nframes,nblocks = cfg.nframes,cfg.nblocks 
    patchsize = cfg.patchsize

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
        
        # -- compute nearest neighbor fields --
        ref_t = nframes//2
        vals,locs = nnf.compute_burst_nnf(dyn_clean,ref_t,patchsize)
        
        # -- run optimization --
        patches = tile_patches(dyn_clean,patchsize).pix
        est_flow = optim.v3.run_batch(patches,evaluator,
                                      nblocks,iterations,
                                      subsizes,K)
