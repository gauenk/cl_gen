
# -- setup project paths --
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")


# -- python imports --
import cv2,tqdm
import numpy as np
import numpy.random as npr
from pathlib import Path

# -- filtering imports --
import pywt
import pywt.data

# -- matplotlib plotting imports --
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# -- project imports --
import cache_io
from pyutils import save_image,images_to_psnrs
from pyutils.vst import anscombe
from patch_search.pixel.bootstrap_numba import fill_weights
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

# -- local imports --
from configs import get_cfg_defaults,setup_exp_cfg,get_exp_cfgs

EXP_PATH = Path("./output/figures/patch_matching/")

def create_quality_plot():
    pass

def execute_experiment(cfg):

    # -- set seed --
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -- load dataset --
    data,loaders = load_image_dataset(cfg)
    image_iter = iter(loaders.tr)
    
    # -- run over images --
    for i_index in range(NUM_BATCHES):

        # -- sample & unpack batch --
        sample = next(image_iter)
        sample_to_cuda(sample)

        dyn_noisy = sample['noisy'] # dynamics and noise
        dyn_clean = sample['burst'] # dynamics and no noise
        static_noisy = sample['snoisy'] # no dynamics and noise
        static_clean = sample['sburst'] # no dynamics and no noise
        flow_gt = sample['flow']
        image_index = sample['index']
        tl_index = sample['tl_index']
        rng_state = sample['rng_state']
        if cfg.noise_params.ntype == "pn":
            dyn_noisy = anscombe.forward(dyn_noisy)

        

    


def main():

    # -- settings --
    cfg = get_cfg_defaults()
    cfg.use_anscombe = True
    cfg.noise_params.ntype = 'g'
    cfg.noise_params.g.std = 25.
    cfg.nframes = 3
    cfg.num_workers = 0
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 10
    cfg.nblocks = 3
    cfg.patchsize = 10
    cfg.gpuid = 1
    cfg.device = f"cuda:{cfg.gpuid}"

    # -- get exps -- 
    experiments,order,egrid = get_exp_cfgs()

    # -- setup cache --
    cache_name = "quality_v_noisy"
    cache_root = EXP_PATH / cache_name
    cache = cache_io.ExpCache(cache_root,cache_name)

    # -- Run Experiments --
    exp_cfgs = []
    for config in tqdm.tqdm(experiments,total=len(experiments)):
        results = cache.load_exp(config)
        uuid = cache.get_uuid(config)
        print(uuid)
        exp_cfg = setup_exp_cfg(cfg,config)
        exp_cfg.uuid = uuid
        exp_cfgs.append(exp_cfg)
        if results is None:
            results = execute_experiment(exp_cfg)
            print(results)
            cache.save_exp(exp_cfg.uuid,config,results)
    records = cache.load_flat_records(experiments)
    print(records)

    # -- init search methods --
    create_quality_plot(cfg,records)

if __name__ == "__main__":
    main()
