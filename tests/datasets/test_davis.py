
from pyutils.testing import get_cfg_defaults,set_seed,convert_keys,is_converted



# -- python --
import cv2
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
from align.xforms import pix_to_flow,align_from_pix,flow_to_pix,align_from_flow
from datasets import load_dataset
# from datasets.kitti import write_burst_kitti_nnf,write_burst_with_flow_kitti_nnf

# from datasets.wrap_image_data import load_image_dataset,load_resample_dataset,sample_to_cuda
SAVE_PATH = Path(f"{settings.ROOT_PATH}/output/tests/datasets/test_davis/")


def get_cfg_defaults():
    cfg = edict()

    # -- frame info --
    cfg.nframes = 1
    cfg.frame_size = [32,32]

    # -- data config --
    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.name = "davis"
    cfg.dataset.dict_loader = True
    cfg.dataset.num_workers = 0
    cfg.set_worker_seed = True
    cfg.batch_size = 1
    cfg.drop_last = {'tr':True,'val':True,'te':True}
    cfg.noise_params = edict({'pn':{'alpha':10.,'std':0},
                              'g':{'std':25.0},'ntype':'g'})
    cfg.dynamic_info = edict()
    cfg.dynamic_info.mode = 'global'
    cfg.dynamic_info.frame_size = cfg.frame_size
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 2
    cfg.dynamic_info.textured = True
    cfg.random_seed = 0

    # -- combo config --
    cfg.nblocks = 5
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

def warp_burst_flow(burst, flows):
    nframes,nimages,ncolor,h,w = burst.shape
    flows = rearrange(flows,'i (h w) tm1 two -> tm1 i h w two',h=h)
    # flows = torch.flip(flows,(-1,))
    burst = rearrange(burst,'t i c h w -> t i h w c')
    wbatch = []
    for i in range(nimages):
        warped = []
        for t in range(nframes):
            if t == nframes//2:
                warped.append(burst[t,i])
                continue
            if t > nframes//2: t_f = t
            else: t_f = t
            img,flow = burst[t,i].numpy(),flows[t_f,i].numpy()
            img = img.astype(np.float32)
            flow = flow.astype(np.float32)
            # flow[...,1] = -flow[...,1]
            wimg = warp_flow(img,-flow)
            wimg = torch.FloatTensor(wimg)
            warped.append(wimg)
        warped = torch.stack(warped)
        wbatch.append(warped)
    wbatch = rearrange(torch.stack(wbatch),'i t h w c -> t i c h w')
    return wbatch
        
def warp_flow(img, flow):
    print("[warp_flow]: img.shape flow.shape ",img.shape,flow.shape)
    print("[warp_flow]: img.dtype flow.dtype ",img.dtype,flow.dtype)
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def test_davis_dataset():
    
    # -- run exp --
    cfg = get_cfg_defaults()
    cfg.random_seed = 123
    nbatches = 20

    # -- set random seed --
    set_seed(cfg.random_seed)	

    # -- load dataset --
    cfg.nframes = 0
    # cfg.frame_size = [128,128]
    cfg.frame_size = None
    cfg.dataset.name = "davis"
    data,loaders = load_dataset(cfg,"dynamic")
    image_iter = iter(loaders.tr)
    sample = next(image_iter)
    print(len(loaders.tr))
    fn = "./davis_example.png"
    save_image(sample['dyn_noisy'],fn,normalize=True,vrange=(0.,1.))

    # -- save path for viz --
    save_dir = SAVE_PATH
    if not save_dir.exists(): save_dir.mkdir(parents=True)
    
    # -- sample data --
    for image_index in range(nbatches):

        # -- sample image --
        index = -1
        # while index != 3233:
        #     sample = next(image_iter)
        #     convert_keys(sample)
        #     index = sample['image_index'][0][0].item()

        sample = next(image_iter)
        # batch_dim0(sample)
        # convert_keys(sample)
        
        # -- extract info --
        noisy = sample['dyn_noisy']
        clean = sample['dyn_clean']
        snoisy = sample['static_noisy']
        sclean = sample['static_clean']
        flow = sample['ref_flow']
        index = sample['index'][0][0].item()
        nframes,nimages,c,h,w = noisy.shape
        mid_pix = h*w//2+2*cfg.nblocks

        # -- print shapes --
        print("-"*50)
        for key,val in sample.items():
            if isinstance(val,list): continue
            print("{}: {}".format(key,val.shape))
        print("-"*50)

        
        print(f"Image Index {index}")

        # -- io info --
        image_dir = save_dir / f"index{index}/"
        if not image_dir.exists(): image_dir.mkdir()

        #
        # -- Compute NNF to Ensure things are OKAY --
        #

        isize = edict({'h':h,'w':w})
        # pad = cfg.patchsize//2 if cfg.patchsize > 1 else 1
        pad = cfg.nblocks//2+1
        psize = edict({'h':h-2*pad,'w':w-2*pad})
        flow_gt = rearrange(flow,'i fm1 h w two -> i (h w) fm1 two')
        pix_gt = flow_to_pix(flow_gt.clone(),nframes,isize=isize)
        def cc(image): return tvF.center_crop(image,(psize.h,psize.w))


        nnf_vals,nnf_pix = compute_burst_nnf(clean,nframes//2,cfg.patchsize)
        shape_str = 't b h w two -> b (h w) t two'
        pix_global = torch.LongTensor(rearrange(nnf_pix[...,0,:],shape_str))
        flow_global = pix_to_flow(pix_global.clone())
        # aligned_gt = warp_burst_flow(clean, flow_global)
        aligned_gt = align_from_pix(clean,pix_gt,cfg.nblocks)
        # isize = edict({'h':h,'w':w})
        # aligned_gt = align_from_flow(clean,flow_global,cfg.nblocks,isize=isize)
        # psnr = compute_aligned_psnr(sclean[[nframes//2]],clean[[nframes//2]],psize)
        psnr = compute_aligned_psnr(sclean,aligned_gt,psize)
        print(f"[GT Alignment] PSNR: {psnr}")


        #
        # -- Save Images to Qualitative Inspect --
        #

        fn = image_dir / "noisy.png"
        save_image(cc(noisy),fn,normalize=True,vrange=None)

        fn = image_dir / "clean.png"
        save_image(cc(clean),fn,normalize=True,vrange=None)
    
        print(cc(sclean).shape)
        fn = image_dir / "diff.png"
        save_image(cc(sclean) - cc(aligned_gt),fn,normalize=True,vrange=None)

        fn = image_dir / "aligned_gt.png"
        save_image(cc(aligned_gt),fn,normalize=True,vrange=None)

