

# -- python --
import random
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch --
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

# -- project --
import settings
from pyutils import save_image
from align import compute_aligned_psnr
from align.nnf import compute_burst_nnf
from patch_search import get_score_function
from align.combo.optim import AlignOptimizer
from align.combo import EvalBlockScores,EvalBootBlockScores
from align.xforms import pix_to_flow,align_from_pix,flow_to_pix
from datasets.wrap_image_data import load_image_dataset,load_resample_dataset,sample_to_cuda


def get_cfg_defaults():
    cfg = edict()

    # -- frame info --
    cfg.nframes = 3
    cfg.frame_size = 32

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "voc"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 2
    cfg.set_worker_seed = True
    cfg.batch_size = 1
    cfg.drop_last = {'tr':True,'val':True,'te':True}
    cfg.noise_params = edict({'pn':{'alpha':10.,'std':0},
                              'g':{'std':25.0},'ntype':'g'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'global'
    cfg.dynamic_info.frame_size = cfg.frame_size
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 1
    cfg.dynamic_info.textured = True
    cfg.random_seed = 0

    # -- combo config --
    cfg.nblocks = 3
    cfg.patchsize = 3
    cfg.score_fxn_name = "bootstrapping"

    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def batch_dim0(sample):
    dim1 = ['burst','noisy','res','clean_burst','sburst','snoisy']    
    skeys = list(sample.keys())
    for field in dim1:
        if not(field in skeys): continue
        sample[field] = sample[field].transpose(1,0)

def convert_keys(sample):
    translate = {'noisy':'dyn_noisy',
                 'burst':'dyn_clean',
                 'snoisy':'static_noisy',
                 'sburst':'static_clean',
                 'flow':'flow_gt',
                 'index':'image_index'}

    for field1,field2 in translate.items():
        sample[field2] = sample[field1]
        del sample[field1]
    return sample

def test_global_dynamics():

    # -- run exp --
    cfg = get_cfg_defaults()
    cfg.random_seed = 123
    nbatches = 20

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load dataset --
    print("load image dataset.")
    data,loaders = load_image_dataset(cfg)
    image_iter = iter(loaders.tr)    

    # -- save path for viz --
    save_dir = Path(f"{settings.ROOT_PATH}/output/tests/datasets/test_global_dynamics/")
    if not save_dir.exists(): save_dir.mkdir(parents=True)
    
    # -- sample data --
    for image_index in range(nbatches):

        # -- sample image --
        sample = next(image_iter)
        # batch_dim0(sample)
        convert_keys(sample)
        
        # -- extract info --
        noisy = sample['dyn_noisy']
        clean = sample['dyn_clean']
        snoisy = sample['static_noisy']
        sclean = sample['static_clean']
        flow = sample['flow_gt']
        index = sample['image_index'][0][0].item()
        nframes,nimages,c,h,w = noisy.shape
        print(f"Image Index {index}")

        # -- io info --
        image_dir = save_dir / f"index{index}/"
        if not image_dir.exists(): image_dir.mkdir()


        #
        # -- Compute NNF to Ensure things are OKAY --
        #

        isize = edict({'h':h,'w':w})
        psize = edict({'h':h-3,'w':w-3})
        flow = repeat(flow,'i fm1 two -> i s fm1 two',s=h*w)
        pix_gt = flow_to_pix(flow.clone(),isize=isize)
        def cc(image): return tvF.center_crop(image,(psize.h,psize.w))

        # -- NNF Global --
        nnf_vals,nnf_pix = compute_burst_nnf(clean,nframes//2,cfg.patchsize)
        shape_str = 't b h w two -> b (h w) t two'
        pix_global = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        flows = pix_to_flow(pix_global)
        aligned_global = align_from_pix(clean,pix_gt,cfg.nblocks)
        psnr = compute_aligned_psnr(clean,aligned_global,psize)
        print(f"[NNF Global] PSNR: {psnr}")

        # -- NNF Local --
        iterations,K,subsizes =0,1,[]
        optim = AlignOptimizer("v3")
        score_fxn_ave = get_score_function("ave")
        eval_ave = EvalBlockScores(score_fxn_ave,"ave",cfg.patchsize,256,None)
        flow = optim.run(clean,cfg.patchsize,eval_ave,
                              cfg.nblocks,iterations,subsizes,K)
        pix_local = flow_to_pix(flow.clone(),isize=isize)
        aligned_local = align_from_pix(clean,pix_gt,cfg.nblocks)
        psnr = compute_aligned_psnr(clean,aligned_local,psize)
        print(f"[NNF Local] PSNR: {psnr}")

        mid_pix = h*w//2+2*cfg.nblocks

        
        # -- remove boundary from pix --
        pixes = {'gt':pix_gt,'global':pix_global,'local':pix_local}
        for field,pix in pixes.items():
            pix_img = rearrange(pix,'i (h w) t two -> (i t) two h w',h=h)
            pix_cc = cc(pix_img)
            pixes[field] = pix_cc

        # -- pairwise diffs --
        field2 = "gt"
        for field1 in pixes.keys():
            if field1 == field2: continue
            delta = pixes[field1] - pixes[field2]
            delta = delta.type(torch.float)
            delta_fn = image_dir / f"delta_{field1}_{field2}.png"
            save_image(delta,delta_fn,normalize=True,vrange=None)
        print(pix_gt[0,mid_pix])
        print(pix_global[0,mid_pix])
        print(pix_local[0,mid_pix])


        #
        # -- Save Images to Qualitative Inspect --
        #

        fn = image_dir / "noisy.png"
        save_image(cc(noisy),fn,normalize=True,vrange=None)

        fn = image_dir / "clean.png"
        save_image(cc(clean),fn,normalize=True,vrange=None)
    
        fn = image_dir / "aligned_global.png"
        save_image(cc(aligned_global),fn,normalize=True,vrange=None)

        fn = image_dir / "aligned_local.png"
        save_image(cc(aligned_local),fn,normalize=True,vrange=None)

