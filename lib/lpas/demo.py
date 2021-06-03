
# -- python imports --
import numpy as np
from PIL import Image
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF
cc = tvF.center_crop

# -- project imports --
from pyutils import save_image,read_image,images_to_psnrs,print_tensor_stats
from n2sim.sim_search import compute_similar_bursts,compute_similar_bursts_analysis
from lpas import lpas_search,lpas_spoof
from datasets.transforms import get_noise_transform,get_dynamic_transform

def main():

    #
    # -- init experiment --
    #

    cfg = edict()
    cfg.gpuid = 1
    cfg.noise_params = edict()
    cfg.noise_params.g = edict()
    # data = load_dataset(cfg)
    torch.manual_seed(143) #131 = 80% vs 20%

    #
    # -- pick our noise --
    #

    # -- gaussian noise --
    # cfg.noise_type = 'g'
    # cfg.noise_params['g']['mean'] = 0.
    # cfg.noise_params['g']['stddev'] = 125.
    # cfg.noise_params.ntype = cfg.noise_type

    # -- poisson noise --
    cfg.noise_type = "pn"
    cfg.noise_params['pn'] = edict()
    cfg.noise_params['pn']['alpha'] = 1.0
    cfg.noise_params['pn']['std'] = 0.0
    cfg.noise_params.ntype = cfg.noise_type

    # -- low-light noise --
    # cfg.noise_type = "qis"
    # cfg.noise_params['qis'] = edict()
    # cfg.noise_params['qis']['alpha'] = 4.0
    # cfg.noise_params['qis']['readout'] = 0.0
    # cfg.noise_params['qis']['nbits'] = 3
    # cfg.noise_params['qis']['use_adc'] = True
    # cfg.noise_params.ntype = cfg.noise_type

    #
    # -- setup the dynamics --
    #

    cfg.nframes = 5
    cfg.frame_size = 350
    cfg.nblocks = 5
    T = cfg.nframes
    
    cfg.dynamic = edict()
    cfg.dynamic.frames = cfg.nframes
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 1
    cfg.dynamic.mode = "global"
    cfg.dynamic.random_eraser = False
    cfg.dynamic.frame_size = cfg.frame_size
    cfg.dynamic.total_pixels = cfg.dynamic.ppf*(cfg.nframes-1)

    # -- setup noise and dynamics --
    noise_xform = get_noise_transform(cfg.noise_params,noise_only=True)
    def null(image): return image
    dynamics_xform = get_dynamic_transform(cfg.dynamic,null)

    # -- sample data --
    image_path = "./data/512-512-grayscale-image-Cameraman.png"
    image = Image.open(image_path).convert("RGB")
    image = image.crop((0,0,cfg.frame_size,cfg.frame_size))
    clean,res,raw,flow = dynamics_xform(image)
    clean = clean[:,None]
    burst = noise_xform(clean+0.5)
    flow = flow[None,:]
    reference = repeat(clean[[T//2]],'1 b c h w -> t b c h w',t=T)
    print("Flow")
    print(flow)

    # -- our method --
    ref_frame = T//2
    nblocks = cfg.nblocks
    method = "simple"
    noise_info = cfg.noise_params
    scores,aligned_simp,dacc_simp = lpas_search(burst,ref_frame,nblocks,
                                                flow,method,clean,noise_info)

    # -- split search --
    ref_frame = T//2
    nblocks = cfg.nblocks
    method = "split"
    noise_info = cfg.noise_params
    scores,aligned_split,dacc_split = lpas_search(burst,ref_frame,nblocks,
                                                  flow,method,clean,noise_info)

    # -- quantitative comparison --
    crop_size = 256
    image1,image2 = cc(aligned_simp,crop_size),cc(reference,crop_size)
    psnrs = images_to_psnrs(image1,image2)
    print("Aligned Simple Method: ",psnrs,dacc_simp.item())
    image1,image2 = cc(aligned_split,crop_size),cc(reference,crop_size)
    psnrs = images_to_psnrs(image1,image2)
    print("Aligned Split Method: ",psnrs,dacc_split.item())

    # -- compute noise 2 sim --
    # T,K = cfg.nframes,cfg.nframes
    # patchsize = 31
    # query = burst[[T//2]]
    # database = torch.cat([burst[:T//2],burst[T//2+1:]])
    # clean_db = clean
    # sim_outputs = compute_similar_bursts_analysis(cfg,query,database,clean_db,K,-1.,
    #                                               patchsize=patchsize,shuffle_k=False,
    #                                               kindex=None,only_middle=False,
    #                                               search_method="l2",db_level="burst")
    # sims,csims,wsims,b_dist,b_indx = sim_outputs

    # -- display images --
    print(aligned_simp.shape)
    print(aligned_split.shape)
    print_tensor_stats("aligned",aligned_simp)
    
    # print(csims.shape)
    save_image(burst,"lpas_demo_burst.png",[-0.5,0.5])
    save_image(clean,"lpas_demo_clean.png")

    save_image(aligned_simp,"lpas_demo_aligned_simp.png")
    save_image(aligned_split,"lpas_demo_aligned_split.png")
    save_image(cc(aligned_simp,crop_size),"lpas_demo_aligned_simp_ccrop.png")
    save_image(cc(aligned_split,crop_size),"lpas_demo_aligned_split_ccrop.png")

    delta_full_simp = aligned_simp - aligned_simp[T//2] 
    delta_full_split = aligned_split - aligned_split[T//2] 
    save_image(delta_full_simp,"lpas_demo_aligned_full_delta_simp.png",[-0.5,0.5])
    save_image(delta_full_split,"lpas_demo_aligned_full_delta_split.png",[-0.5,0.5])

    delta_cc_simp = cc(delta_full_simp,crop_size)
    delta_cc_split = cc(delta_full_split,crop_size)
    save_image(delta_full_simp,"lpas_demo_aligned_cc_delta_simp.png")
    save_image(delta_full_split,"lpas_demo_aligned_cc_delta_split.png")
    
    top = 75
    size = 64
    simp = tvF.crop(aligned_simp,top,200,size,size)
    split = tvF.crop(aligned_split,top,200,size,size)
    print_tensor_stats("delta",simp)
    save_image(simp,"lpas_demo_aligned_simp_inspect.png")
    save_image(split,"lpas_demo_aligned_split_inspect.png")


    delta_simp = simp - simp[T//2]
    delta_split = split - split[T//2]
    print_tensor_stats("delta",delta_simp)
    save_image(delta_simp,"lpas_demo_aligned_simp_inspect_delta.png",[-1,1.])
    save_image(delta_split,"lpas_demo_aligned_split_inspect_delta.png",[-1,1.])

    # save_image(csims,"lpas_demo_sim.png")
