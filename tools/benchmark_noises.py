""""

Create a histogram of center pixels given the K-means clustering

"""

# -- python imports --
import sys,copy
sys.path.append("./lib")
import pandas as pd
from pathlib import Path
from einops import rearrange,reduce
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

# -- project imports --
from datasets import load_dataset
from pyutils.misc import mse_to_psnr
from n2sim.main import get_main_config
from n2sim.sim_search import compute_similar_bursts_analysis,kIndexPermLMDB

def compute_percent_std(pix_diff,score=1.65):
    perc = []
    K,D = pix_diff.shape
    for k in range(K):
        std = torch.std(pix_diff[k])
        idx = torch.where(torch.abs(pix_diff[k]) > score*std)[0]
        O = float(len(idx))
        perc.append((O / D)*100.)
    return perc

def create_bw_image_from_flat_idx(flat_idx,H,W):
    B,S = flat_idx.shape
    flat_images = torch.zeros( (B,H*W) )
    for b in range(B): flat_images[b,flat_idx[b]] = 1.
    images = rearrange(flat_images,'b (h w) -> b 1 h w',h=H)
    images = images.repeat(1,3,1,1)
    return images
    

def flatten_patches(image,ps=3):
    unfold = nn.Unfold(ps,1,0,1)
    image_pad = F.pad(image,(ps//2,ps//2,ps//2,ps//2),mode='reflect')
    patches = unfold(image_pad)
    patches = rearrange(patches,'b (c ps1 ps2) r -> b r (ps1 ps2 c)',ps1=ps,ps2=ps)
    patches = patches.contiguous()
    return patches

def get_smooth_patches(image,ps=3,S=1000):
    return get_sobel_patches(image,ps,S,False)

def get_nonsmooth_patches(image,ps=3,S=1000):
    return get_sobel_patches(image,ps,S,True)

def get_sobel_patches(image,ps,S,descend):
    # -- create patches --
    patches = flatten_patches(image,ps=ps)

    # -- get sobel filter to detect rough spots --
    sobel = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_t = sobel.t()
    sobel = sobel.reshape(1,-1).repeat(3,1).reshape(-1)
    sobel_t = sobel_t.reshape(1,-1).repeat(3,1).reshape(-1)
    weights = torch.stack([sobel,sobel_t],dim=0)

    # -- compute smoothness of each patch --
    smoothness = torch.mean(torch.abs(torch.einsum('brl,cl->brc',patches,weights)),dim=-1)
    B,D = smoothness.shape
    nonsmooth = []
    for b in range(B):
        values,indices = torch.sort(smoothness[b],descending=descend)
        nonsmooth_idx = indices[:S]
        nonsmooth.append(nonsmooth_idx)
    nonsmooth = torch.stack(nonsmooth,dim=0)

    return nonsmooth

def create_noise_level_grid(cfg):
    noise_settings = []

    # -- gaussian noise --
    noise_type = 'g'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = 0.01
    ns['name'] = f"g-{ns['stddev']}".replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- gaussian noise --
    noise_type = 'g'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = 25.
    ns['name'] = f"g-{ns['stddev']}".replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)
    
    # -- heavy gaussian noise --
    noise_type = 'g'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['stddev'] = 75.
    ns['name'] = f"g-{ns['stddev']}".replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- heteroskedastic gaussian noise --
    noise_type = 'hg'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['read'] = 25.
    ns['shot'] = 15.
    ns['name'] = f"hg-{ns['read']}-{ns['shot']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- heavy heteroskedastic gaussian noise --
    noise_type = 'hg'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['read'] = 75.
    ns['shot'] = 25.
    ns['name'] = f"hg-{ns['read']}-{ns['shot']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- poisson noise --
    noise_type = 'pn'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 40.
    ns['std'] = 0.
    ns['name'] = f"pn-{ns['alpha']}-{ns['std']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- poisson noise --
    noise_type = 'pn'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 4.
    ns['std'] = 0.
    ns['name'] = f"pn-{ns['alpha']}-{ns['std']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- poisson noise --
    noise_type = 'pn'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 1.
    ns['std'] = 0.
    ns['name'] = f"pn-{ns['alpha']}-{ns['std']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- poisson + gaussian noise --
    noise_type = 'pn'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 40.
    ns['std'] = 15.
    ns['name'] = f"pn-{ns['alpha']}-{ns['std']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- poisson + gaussian noise --
    noise_type = 'pn'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 4.
    ns['std'] = 15.
    ns['name'] = f"pn-{ns['alpha']}-{ns['std']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- poisson + gaussian noise --
    noise_type = 'pn'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 4.
    ns['std'] = 1.
    ns['name'] = f"pn-{ns['alpha']}-{ns['std']}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- adc + poisson noise --
    noise_type = 'qis'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 40.
    ns['readout'] = 0.
    ns['nbits'] = 8
    ns['name'] = f"qis-{int(ns['alpha'])}-{int(ns['readout'])}-{int(ns['nbits'])}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- adc + poisson noise --
    noise_type = 'qis'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 40.
    ns['readout'] = 25.
    ns['nbits'] = 8
    ns['name'] = f"qis-{int(ns['alpha'])}-{int(ns['readout'])}-{int(ns['nbits'])}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- adc + poisson noise --
    noise_type = 'qis'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 4.
    ns['readout'] = 25.
    ns['nbits'] = 3
    ns['name'] = f"qis-{int(ns['alpha'])}-{int(ns['readout'])}-{int(ns['nbits'])}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)


    # -- adc + poisson noise --
    noise_type = 'qis'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 1.
    ns['readout'] = 15.
    ns['nbits'] = 3
    ns['name'] = f"qis-{int(ns['alpha'])}-{int(ns['readout'])}-{int(ns['nbits'])}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- adc + poisson noise --
    noise_type = 'qis'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 4.
    ns['readout'] = 0.
    ns['nbits'] = 3
    ns['name'] = f"qis-{int(ns['alpha'])}-{int(ns['readout'])}-{int(ns['nbits'])}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    # -- adc + poisson noise --
    noise_type = 'qis'
    ns = copy.deepcopy(cfg.noise_params[noise_type])
    ns['ntype'] = noise_type
    ns['alpha'] = 1.
    ns['readout'] = 0.
    ns['nbits'] = 3
    ns['name'] = f"qis-{int(ns['alpha'])}-{int(ns['readout'])}-{int(ns['nbits'])}"
    ns['name'] = ns['name'].replace(".","p")
    ns = edict(ns)
    noise_settings.append(ns)

    return noise_settings

def set_noise_setting(cfg,ns):
    cfg.noise_params.ntype = ns.ntype
    if ns.ntype == "g":
        cfg.noise_params[ns.ntype]['stddev'] = ns.stddev
    elif ns.ntype == "hg":
        cfg.noise_params[ns.ntype]['read'] = ns.read
        cfg.noise_params[ns.ntype]['shot'] = ns.shot
    elif ns.ntype == "pn":
        cfg.noise_params[ns.ntype]['alpha'] = ns.alpha
        cfg.noise_params[ns.ntype]['std'] = ns.std
    elif ns.ntype == "qis":
        cfg.noise_params[ns.ntype]['alpha'] = ns.alpha
        cfg.noise_params[ns.ntype]['readout'] = ns.readout
        cfg.noise_params[ns.ntype]['nbits'] = ns.nbits
    else:
        raise ValueError(f"Uknown noise type [{ns.ntype}]")
    return ns.name

def main():

    # -- init --
    cfg = get_main_config()
    cfg.gpuid = 0
    cfg.batch_size = 1
    cfg.N = 2
    cfg.num_workers = 0
    cfg.dynamic.frames = cfg.N
    cfg.rot = edict()
    cfg.rot.skip = 0 # big gap between 2 and 3.

    # -- dynamics --
    cfg.dataset.name = "rots"
    cfg.dataset.load_residual = True
    cfg.dynamic.frame_size = 256
    cfg.frame_size = cfg.dynamic.frame_size
    cfg.dynamic.ppf = 0
    cfg.dynamic.total_pixels = cfg.N*cfg.dynamic.ppf
    torch.cuda.set_device(cfg.gpuid)

    # -- sim params --
    K = 10
    patchsize = 9
    db_level = "frame"
    search_method = "l2"
    # database_str = f"burstAll"
    database_idx = 1
    database_str = "burst{}".format(database_idx)

    # -- grab grids for experiments --
    noise_settings = create_noise_level_grid(cfg)
    # sim_settings = create_sim_grid(cfg)
    # motion_settings = create_motion_grid(cfg)
    
    for ns in noise_settings:

        # -=-=-=-=-=-=-=-=-=-=-
        #     loop params
        # -=-=-=-=-=-=-=-=-=-=-
        noise_level = 0.
        noise_type = ns.ntype
        noise_str = set_noise_setting(cfg,ns)
        
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #    create path for results
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        path_args = (K,patchsize,cfg.batch_size,cfg.N,noise_str,
                     database_str,db_level,search_method)
        base = Path(f"output/benchmark_noise_types/{cfg.dataset.name}")
        root = Path(base / "k{}_ps{}_b{}_n{}_{}_db-{}_sim-{}-{}".format(*path_args))
        print(f"Writing to {root}")
        if root.exists(): print("Running Experiment Again.")
        else: root.mkdir(parents=True)

        # -=-=-=-=-=-=-
        #   dataset
        # -=-=-=-=-=-=-
        data,loader = load_dataset(cfg,'dynamic')
        if cfg.dataset.name == "voc":
            sample = next(iter(loader.tr))
        else:
            sample = data.tr[0]

        # -- load sample --
        burst,raw_img,res = sample['burst'],sample['clean']-0.5,sample['res']
        kindex_ds = kIndexPermLMDB(cfg.batch_size,cfg.N)    
        N,B,C,H,W = burst.shape
        if 'clean_burst' in sample: clean = sample['clean_burst']-0.5
        else: clean = burst - res
        if noise_type in ["qis","pn"]: tvF.rgb_to_grayscale(clean,3)
        # burst = tvF.rgb_to_grayscale(burst,3)
        # raw_img = tvF.rgb_to_grayscale(raw_img,3)
        # clean = tvF.rgb_to_grayscale(clean,3)
        
        # -- temp (delete me soon) --
        search_rot_grid = np.linspace(.3,.32,100)
        losses = np.zeros_like(search_rot_grid)
        for idx,angle in enumerate(search_rot_grid):
            save_alpha_burst = 0.5*burst[0] + 0.5*tvF.rotate(burst[1],angle)
            losses[idx] = F.mse_loss(save_alpha_burst,burst[0]).item()
        min_arg = np.argmin(losses)
        angle = search_rot_grid[min_arg]

        ref_img = tvF.rotate(burst[1],angle)
        shift_grid = np.linspace(-20,20,40-1).astype(np.int)
        losses = np.zeros_like(shift_grid).astype(np.float)
        for idx,shift in enumerate(shift_grid):
            save_alpha_burst = 0.5*burst[0] + 0.5*torch.roll(ref_img,shift,-2)
            losses[idx] = F.mse_loss(save_alpha_burst,burst[0]).item()
        min_arg = np.argmin(losses)
        shift = shift_grid[min_arg]

        # -- run search --
        kindex = kindex_ds[0]
        database = None
        if database_str == f"burstAll":
            database = burst
            clean_db = clean
        else:
            database = burst[[database_idx]]
            clean_db = clean[[database_idx]]
        query = burst[[0]]
        sim_outputs = compute_similar_bursts_analysis(cfg,query,database,clean_db,K,
                                                      patchsize=patchsize,
                                                      shuffle_k=False,kindex=kindex,
                                                      only_middle=cfg.sim_only_middle,
                                                      db_level=db_level,
                                                      search_method=search_method,
                                                      noise_level=noise_level/255.)
        sims,csims,wsims,b_dist,b_indx = sim_outputs

        # -- save images --
        fs = cfg.frame_size
        save_K = 1
        save_sims = rearrange(sims[:,:,:save_K],'n b k1 c h w -> (n b k1) c h w')
        save_csims = rearrange(csims[:,:,:save_K],'n b k1 c h w -> (n b k1) c h w')
        save_cdelta = clean[0] - save_csims[0]
        save_alpha_burst = 0.5*burst[0] + 0.5*torch.roll(tvF.rotate(burst[1],angle),shift,-2)
        
        save_burst = rearrange(burst,'n b c h w -> (b n) c h w')
        save_clean = rearrange(clean,'n b c h w -> (b n) c h w')
        save_b_dist = rearrange(b_dist[:,:,:save_K],'n b k1 h w -> (n b k1) 1 h w')
        save_b_indx = rearrange(b_indx[:,:,:save_K],'n b k1 h w -> (n b k1) 1 h w')
        save_b_indx = torch.abs( torch.arange(fs*fs).reshape(fs,fs) - save_b_indx ).float()
        save_b_indx /= (torch.sum(save_b_indx)+1e-16)
        tv_utils.save_image(save_sims,root/'sims.png',nrow=B,normalize=True,range=(-0.5,0.5))
        tv_utils.save_image(save_csims,root/'csims.png',nrow=B,normalize=True,range=(-0.5,0.5))
        tv_utils.save_image(save_cdelta,root/'cdelta.png',nrow=B,normalize=True,range=(-0.5,0.5))
        tv_utils.save_image(save_clean,root/'clean.png',nrow=N,normalize=True,range=(-0.5,0.5))
        tv_utils.save_image(save_burst,root/'burst.png',nrow=N,normalize=True,range=(-0.5,0.5))
        tv_utils.save_image(save_b_dist,root/'b_dist.png',nrow=B,normalize=True)
        tv_utils.save_image(raw_img,root/'raw.png',nrow=B,normalize=True)
        tv_utils.save_image(save_b_indx,root/'b_indx.png',nrow=B,normalize=True)
        tv_utils.save_image(save_alpha_burst,root/'alpha_burst.png',nrow=B,normalize=True)

        # -- save top K patches at location --
        b = 0
        ref_img = clean[0,b]
        ps,fs = patchsize,cfg.frame_size
        xx,yy = np.mgrid[32:48,48:64]
        xx,yy = xx.ravel(),yy.ravel()
        clean_pad = F.pad(clean[database_idx,[b]],(ps//2,ps//2,ps//2,ps//2),mode='reflect')[0]
        patches = []
        for x,y in zip(xx,yy):
            gt_patch = tvF.crop(ref_img,x-ps//2,y-ps//2,ps,ps)
            patches_xy = [gt_patch]
            for k in range(save_K):
                indx = b_indx[0,0,k,x,y]
                xp,yp = (indx // fs)+ps//2, (indx % fs )+ps//2
                t,l = xp-ps//2,yp-ps//2
                clean_patch = tvF.crop(clean_pad,t,l,ps,ps)
                patches_xy.append(clean_patch)
                pix_diff = F.mse_loss(gt_patch[:,ps//2,ps//2],clean_patch[:,ps//2,ps//2]).item()
                pix_diff_img = pix_diff * torch.ones_like(clean_patch)
                patches_xy.append(pix_diff_img)
            patches_xy = torch.stack(patches_xy,dim=0)
            patches.append(patches_xy)
        patches = torch.stack(patches,dim=0)
        R = patches.shape[1]
        patches = rearrange(patches,'l k c h w -> (l k) c h w')
        fn = f"patches_{b}.png"
        tv_utils.save_image(patches,root/fn,nrow=R,normalize=True)
        
        # -- stats about distance --
        mean_along_k = reduce(b_dist,'n b k1 h w -> k1','mean')
        std_along_k = torch.std(b_dist,dim=(0,1,3,4))
        fig,ax = plt.subplots(figsize=(8,8))
        R = mean_along_k.shape[0]
        ax.errorbar(np.arange(R),mean_along_k,yerr=std_along_k)
        plt.savefig(root / "distance_stats.png",dpi=300)
        plt.clf()
        plt.close("all")
    
        # -- psnr between 1st neighbor and clean --
        psnrs = pd.DataFrame({"b":[],"k":[],"psnr":[],'crop200_psnr':[]})
        for b in range(B):
            for k in range(K):

                # -- psnr --
                crop_raw = clean[0,b]
                crop_cmp = csims[0,b,k]
                rc_mse = F.mse_loss(crop_raw,crop_cmp,reduction='none').reshape(1,-1)
                rc_mse = torch.mean(rc_mse,1).numpy() + 1e-16
                psnr_bk = np.mean(mse_to_psnr(rc_mse))
                print(psnr_bk)
                
                # -- crop psnr --
                crop_raw = tvF.center_crop(clean[0,b],200)
                crop_cmp = tvF.center_crop(csims[0,b,k],200)
                rc_mse = F.mse_loss(crop_raw,crop_cmp,reduction='none').reshape(1,-1)
                rc_mse = torch.mean(rc_mse,1).numpy() + 1e-16
                crop_psnr = np.mean(mse_to_psnr(rc_mse))
                # if np.isinf(psnr_bk): psnr_bk = 50.
                psnrs = psnrs.append({'b':b,'k':k,'psnr':psnr_bk,'crop200_psnr':crop_psnr},
                                     ignore_index=True)
        # psnr_ave = np.mean(psnrs)
        # psnr_std = np.std(psnrs)
        # print( "PSNR: %2.2f +/- %2.2f" % (psnr_ave,psnr_std) )
        psnrs = psnrs.astype({'b':int,'k':int,'psnr':float,'crop200_psnr':float})
        psnrs.to_csv(root/"psnrs.csv",sep=",",index=False)

if __name__ == "__main__":
    main()
