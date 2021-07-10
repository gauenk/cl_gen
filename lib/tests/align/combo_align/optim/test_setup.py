

from easydict import EasyDict as edict

import align.nnf
import align.combo_align.optim
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def config():
    cfg = edict()
    cfg = None
    cfg.random_seed = 123
    raise NotImplemented("")
    return cfg

def score_function(name):
    raise NotImplemented("")

def setup():

    # -- get config --
    cfg = config()

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load dataset --
    data,loaders = load_image_dataset(cfg)
    image_iter = iter(loaders.tr)    

    # -- get score function --
    score_fxn = get_score_function("bootstrap")

    # -- create evaluator
    iterations = 3
    K = 3
    subsizes = [3,2,2,2,2,2]
    evaluator = align.eval.EvalBlockScores(score_fxn,100,None)

    # -- iterate over images --
    NUM_BATCHES = 2
    for image_bindex in tqdm(range(NUM_BATCHES),leave=False):

        # -- sample & unpack batch --
        sample = next(image_batch_iter)
        sample_to_cuda(sample)
        dyn_noisy = sample['noisy'] # dynamics and noise
        dyn_clean = sample['burst'] # dynamics and no noise
        static_noisy = sample['snoisy'] # no dynamics and noise
        static_clean = sample['sburst'] # no dynamics and no noise
        flow = sample['flow']
        
        # -- compute nearest neighbor fields --
        nnfs = []
        ref_img = dyn_clean[nframes//2]
        patchsize = cfg.patchsize
        for t in range(nframes):
            if t == nframes//2: continue
            img = dyn_clean[t]
            vals,locs = nnf.compute_nnf(ref_img,img,patchsize)
            nnfs.append(locs)
        
        # -- run optimization --
        patches = tile_patches(dyn_clean)
        est_flow = optim.v3.run(patches,evaluator,
                                cfg.nframes,cfg.nblocks,
                                iterations,subsizes,K)

        # -- compare with nnf --
        

# def run(patches,evaluator,
#         nframes,nblocks,
#         iterations,
#         subsizes,K):

