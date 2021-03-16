""""

Create a histogram of center pixels given the K-means clustering

"""

# -- python imports --
import sys
sys.path.append("./lib")
from pathlib import Path
from einops import rearrange,reduce
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

# -- project imports --
from datasets import load_dataset
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
        values,indices =torch.sort(smoothness[b],descending=descend)
        nonsmooth_idx = indices[:S]
        nonsmooth.append(nonsmooth_idx)
    nonsmooth = torch.stack(nonsmooth,dim=0)

    return nonsmooth

def main():

    cfg = get_main_config()
    cfg.gpuid = 0
    cfg.batch_size = 10
    cfg.N = 8
    cfg.num_workers = 0
    cfg.dynamic.frames = cfg.N
    cfg.noise_params['g']['stddev'] = 200.
    cfg.dataset.load_residual = True
    cfg.dynamic.frame_size = 128
    cfg.dynamic.ppf = 2
    cfg.dynamic.total_pixels = cfg.N*cfg.dynamic.ppf
    torch.cuda.set_device(cfg.gpuid)
    data,loader = load_dataset(cfg,'dynamic')
    train_iter = iter(loader.tr)

    K = 30
    patchsize = 15
    noise_level = int(cfg.noise_params['g']['stddev'])
    sim_mode = "burst"
    # sim_str = "zc"
    search_method = "w"
    
    root = Path("output/n2sim/sim_search_analysis/k{}_ps{}_g{}_sim-{}-{}".format(K,patchsize,noise_level,sim_mode,search_method))
    print(f"Writing to {root}")
    if not root.exists(): root.mkdir(parents=True)

    # -- load sample --
    sample = next(train_iter)
    burst,raw_img,res = sample['burst'],sample['clean']-0.5,sample['res']
    kindex_ds = kIndexPermLMDB(cfg.batch_size,cfg.N)    
    N,B,C,H,W = burst.shape
    clean = burst - res

    # -- run search --
    kindex = kindex_ds[0]
    sim_outputs = compute_similar_bursts_analysis(cfg,burst,clean,K,
                                                  patchsize=patchsize,
                                                  shuffle_k=False,kindex=kindex,
                                                  only_middle=cfg.sim_only_middle,
                                                  sim_mode=sim_mode,
                                                  search_method=search_method,
                                                  noise_level=noise_level/255.)
    sims,csims,wsims,b_dist,b_indx = sim_outputs
    print(b_dist[0,0,:3,:3,:3])
    print(b_dist[0,0,0,:30,0])

    # -- save images --
    save_K = 10
    save_sims = rearrange(sims[:,:,:save_K],'n b k1 c h w -> (n b k1) c h w')
    save_burst = rearrange(burst,'n b c h w -> (b n) c h w')
    save_clean = rearrange(clean,'n b c h w -> (b n) c h w')
    save_b_dist = rearrange(b_dist[:,:,:save_K],'n b k1 h w -> (n b k1) 1 h w')
    tv_utils.save_image(save_sims,root/'sims.png',nrow=B,normalize=True,range=(-0.5,0.5))
    tv_utils.save_image(save_clean,root/'clean.png',nrow=N,normalize=True,range=(-0.5,0.5))
    tv_utils.save_image(save_burst,root/'burst.png',nrow=N,normalize=True,range=(-0.5,0.5))
    tv_utils.save_image(save_b_dist,root/'b_dist.png',nrow=B,normalize=True)

    # -- stats about distance --
    mean_along_k = reduce(b_dist,'n b k1 h w -> k1','mean')
    std_along_k = torch.std(b_dist,dim=(0,1,3,4))
    fig,ax = plt.subplots(figsize=(8,8))
    ax.errorbar(np.arange(K),mean_along_k,yerr=std_along_k)
    plt.savefig(root / "distance_stats.png",dpi=300)
    plt.clf()
    plt.close("all")


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #         Statistics about middle pixel differences
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- stats about middle pixel diff --
    print("About to make the boxplot.... Takes some time...")
    pix_diff = raw_img.unsqueeze(1).repeat(1,K,1,1,1) - csims[0]
    pix_diff = rearrange(pix_diff,'b k c h w -> k (b c h w)')
    fig,ax = plt.subplots(figsize=(8,8))
    ax.boxplot([pix_diff[k] for k in range(pix_diff.shape[0])])
    plt.savefig(root / "mid_pix_stats.png",dpi=300)
    plt.clf()
    plt.close("all")
    print("Completed boxplot.")

    # -- plot percent of samples in the tail --
    pix_diff = raw_img.unsqueeze(1).repeat(1,K,1,1,1) - csims[0]
    pix_diff = rearrange(pix_diff,'b k c h w -> k (b c h w)')
    std = torch.std(pix_diff,dim=1)
    tail_info = compute_percent_std(pix_diff,score=1.65)
    fig,ax = plt.subplots(figsize=(8,8))
    ax.scatter(std,tail_info,c=np.arange(std.shape[0]))
    plt.savefig(root / "tail_info_mid_pix_stats.png",dpi=300)
    plt.clf()
    plt.close("all")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #         Statistics about residual pixel differences
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- stats about residual pixel diff --
    plot_K = K
    pix_diff = raw_img.unsqueeze(1).repeat(1,K,1,1,1) - sims[0,:,1:]
    pix_diff = rearrange(pix_diff,'b k c h w -> k (b c h w)')
    fig,ax = plt.subplots(figsize=(8,8))
    ax.boxplot([pix_diff[k] for k in range(pix_diff.shape[0])])
    plt.savefig(root / "mid_res_stats.png",dpi=300)
    plt.clf()
    plt.close("all")
    print("Completed boxplot.")

    # -- plot percent of samples in the tail --
    pix_diff = raw_img.unsqueeze(1).repeat(1,K,1,1,1) - sims[0,:,1:]
    pix_diff = rearrange(pix_diff,'b k c h w -> k (b c h w)')
    std = torch.std(pix_diff,dim=1)
    tail_info = compute_percent_std(pix_diff,score=1.65)
    fig,ax = plt.subplots(figsize=(8,8))
    ax.scatter(std,tail_info,c=np.arange(std.shape[0]))
    plt.savefig(root / "tail_info_mid_res_stats.png",dpi=300)
    plt.clf()
    plt.close("all")

    # -- histograms --
    # pix_diff = raw_img.unsqueeze(1).repeat(1,K,1,1,1) - csims[0]
    # pix_diff = rearrange(pix_diff,'b k c h w -> k (b c h w)')
    # fig,ax = plt.subplots(figsize=(8,8))
    # for k in range(1): ax.hist(pix_diff[k])
    # plt.savefig(root / "mid_pix_hist.png",dpi=300)
    # plt.clf()
    # plt.close("all")

    # mean_along_k = torch.mean(pix_diff,dim=(0,2,3,4))
    # std_along_k = torch.std(pix_diff,dim=(0,2,3,4))
    # fig,ax = plt.subplots(figsize=(8,8))
    # ax.errorbar(np.arange(K),mean_along_k,yerr=std_along_k)
    # plt.savefig(root / "mid_pix_stats.png",dpi=300)
    # plt.clf()
    # plt.close("all")
    # print("Stats about Middle Pixel Difference")
    # print(mean_along_k,std_along_k)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #
    #      sort via w distance
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # wdist = (b_dist - noise_level/255.)**2
    # print(b_dist.shape,b_indx.shape)
    # c_pix_diff = raw_img.unsqueeze(1).repeat(1,K,1,1,1) - csims[0]
    # c_pix_diff = raw_img.unsqueeze(1).repeat(1,K,1,1,1) - csims[0]



    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #
    #      stats about middle pixel diff for Nonsmooth Patches 
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # -- compute pixel difference --
    pix_diff = raw_img.unsqueeze(1).repeat(1,K,1,1,1) - csims[0]
    pix_diff = pix_diff

    # -- get nonsmooth regions --
    nonsmooth_idx = get_nonsmooth_patches(raw_img)

    # -- visualize nonsmooth indices --
    ns_img = create_bw_image_from_flat_idx(nonsmooth_idx,H,W)
    save_img = torch.cat([raw_img,ns_img],dim=0)
    tv_utils.save_image(save_img,root / "nonsmooth_image.png",nrow=B,normalize=True,range=(0,1))

    # -- flatten --
    pix_diff = rearrange(pix_diff,'b k c h w -> (b k) c h w')
    pix_diff = flatten_patches(pix_diff,ps=3)    
    pix_diff = rearrange(pix_diff,'(b k) l r -> b k l r',b=B)

    # -- index each flattened patch --
    nonsmooth_pix = []
    for b in range(B):
        nonsmooth_pix.append(pix_diff[b,:,nonsmooth_idx[b]])
    nonsmooth_pix = torch.stack(nonsmooth_pix,dim=0)

    mean_along_k = torch.mean(nonsmooth_pix,dim=(0,2,3))
    std_along_k = torch.std(nonsmooth_pix,dim=(0,2,3))
    fig,ax = plt.subplots(figsize=(8,8))
    ax.errorbar(np.arange(K),mean_along_k,yerr=std_along_k)
    plt.savefig(root / "nonsmooth_mid_pix_stats.png",dpi=300)
    plt.clf()
    plt.close("all")

    

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #
    #      stats about middle pixel diff for Smooth Patches 
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # -- compute pixel difference --
    pix_diff = raw_img.unsqueeze(1).repeat(1,K,1,1,1) - csims[0]
    pix_diff = pix_diff

    # -- get smooth regions --
    smooth_idx = get_smooth_patches(raw_img)

    # -- visualize smooth indices --
    ns_img = create_bw_image_from_flat_idx(smooth_idx,H,W)
    save_img = torch.cat([raw_img,ns_img],dim=0)
    tv_utils.save_image(save_img,root / "smooth_image.png",nrow=B,normalize=True,range=(0,1))

    # -- flatten --
    pix_diff = rearrange(pix_diff,'b k c h w -> (b k) c h w')
    pix_diff = flatten_patches(pix_diff,ps=3)    
    pix_diff = rearrange(pix_diff,'(b k) l r -> b k l r',b=B)

    # -- index each flattened patch --
    smooth_pix = []
    for b in range(B):
        smooth_pix.append(pix_diff[b,:,smooth_idx[b]])
    smooth_pix = torch.stack(smooth_pix,dim=0)

    mean_along_k = torch.mean(smooth_pix,dim=(0,2,3))
    std_along_k = torch.std(smooth_pix,dim=(0,2,3))
    fig,ax = plt.subplots(figsize=(8,8))
    ax.errorbar(np.arange(K),mean_along_k,yerr=std_along_k)
    plt.savefig(root / "smooth_mid_pix_stats.png",dpi=300)
    plt.clf()
    plt.close("all")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #
    #                stats about residuals
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    pix_diff = raw_img.unsqueeze(1).repeat(1,K+1,1,1,1) - sims[0]
    mean_along_k = torch.mean(pix_diff,dim=(0,2,3,4))
    std_along_k = torch.std(pix_diff,dim=(0,2,3,4))
    fig,ax = plt.subplots(figsize=(8,8))
    ax.errorbar(np.arange(K+1),mean_along_k,yerr=std_along_k)
    plt.savefig(root / "sim_residuals_stats.png",dpi=300)
    plt.clf()
    plt.close("all")

    pix_diff = raw_img - burst[N//2]
    pix_diff = pix_diff
    mean = torch.mean(pix_diff)
    std = torch.std(pix_diff)
    print("Burst Residual Statistics: %2.2f +/- %2.2f" % (mean,std))



if __name__ == "__main__":
    main()
