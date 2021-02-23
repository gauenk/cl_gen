# -- python imports --
import numpy as np
import numpy.random as npr
from einops import rearrange, repeat, reduce

# -- faiss imports --
import faiss
import faiss.contrib.torch_utils

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils.timer import Timer

# -- [local] image comparisons --
from .nearest_search import search_raw_array_pytorch
from .prepare_bsd400_lmdb import compute_sim_images,shift_concat_image

def compare_sim_images_methods(cfg,burst,K,patchsize=3):
    b = 0
    n = 0
    ps = patchsize
    img = burst[n,b]
    single_burst = img.unsqueeze(0).unsqueeze(0)

    img_rs = rearrange(img,'c h w -> h w c').cpu()
    t_1 = Timer()
    t_1.tic()
    sim_burst_v1 = rearrange(torch.tensor(compute_sim_images(img_rs, patchsize, K)),'k h w c -> k c h w')
    t_1.toc()

    t_2 = Timer()
    t_2.tic()
    sim_burst_v2 = compute_similar_bursts(cfg,single_burst,K,patchsize=3)[0,0,1:]
    t_2.toc()

    print(t_1,t_2)
    
    print("v1",sim_burst_v1.shape)
    print("v2",sim_burst_v2.shape)
    for k in range(K):
        print("mse-v1-{}".format(k),F.mse_loss(sim_burst_v1[k].cpu(),img.cpu()))
        print("mse-v2-{}".format(k),F.mse_loss(sim_burst_v2[k].cpu(),img.cpu()))
        print("mse-{}".format(k),F.mse_loss(sim_burst_v1[k].cpu(),sim_burst_v2[k].cpu()))

def compare_sim_patches_methods(cfg,burst,K,patchsize=3):
    b = 0
    n = 0
    ps = patchsize
    img = burst[n,b]
    print("img",img.shape)

    img_rs = rearrange(img,'c h w -> h w c').cpu()
    patches_v1 = shift_concat_image(img_rs, patchsize)
    [H, W, D, C] = patches_v1.shape
    patches_v1 = patches_v1.reshape([H * W, D * C]).astype(np.float32)
    patches_v1 = torch.tensor(patches_v1)

    
    img_pad = F.pad(img.unsqueeze(0),(ps//2,ps//2,ps//2,ps//2),mode='reflect')
    print("pad",img_pad.shape)
    unfold = nn.Unfold(patchsize,1,0,1)
    patches_v2 = rearrange(unfold(img_pad),'1 l r -> r l')
    
    print("v1",patches_v1.shape)
    print("v2",patches_v2.shape)
    print("l2",F.mse_loss(patches_v1,patches_v2.cpu()))

    patches_v2 = rearrange(patches_v2,'r (c ps1 ps2) -> r ps1 ps2 c',ps1=ps,ps2=ps)
    patches_v2[...,ps//2,ps//2,:] = 0
    patches_v2 = rearrange(patches_v2,'r ps1 ps2 c -> r (ps1 ps2 c)')

    print("l2-zmid",F.mse_loss(patches_v1,patches_v2.cpu()))

    print(patches_v1[0].reshape(3,3,3))
    print(patches_v2[0].reshape(3,3,3))

    
def compute_similar_bursts_n2sim(cfg,burst,K,patchsize=3):
    N,B = burst.shape[:2]
    sim_burst = []
    for b in range(B):
        sim_frames = []
        for n in range(N):
            image = rearrange(burst[n,b].cpu(),'c h w -> h w c')
            sim_images = compute_sim_images(image, patchsize, K, img_ori=None) 
            sim_images = rearrange(sim_images,'k h w c -> k c h w')
            sim_images = torch.tensor(sim_images)
            sim_frames.append(sim_images)
        sim_frames = torch.stack(sim_frames,dim=0)
        sim_burst.append(sim_frames)
    sim_burst = torch.stack(sim_burst,dim=1).to(cfg.gpuid)
    # for k in range(K):
    #     print(k,F.mse_loss(sim_burst[:,:,k],burst))
    return sim_burst

def compute_similar_bursts(cfg,burst,K,patchsize=3,shuffle_k=False):
    """
    params: burst shape: [N, B, C, H, W]
    """

    # -- init shapes --
    ps = patchsize
    N,B,C,H,W = burst.shape

    # -- tile patches --
    unfold = nn.Unfold(patchsize,1,0,1)
    burst_pad = rearrange(burst,'n b c h w -> (b n) c h w')
    burst_pad = F.pad(burst_pad,(ps//2,ps//2,ps//2,ps//2),mode='reflect')
    # print("pad",burst_pad.shape)
    patches = unfold(burst_pad)

    # -- remove center pixels from patches for search --
    patches = rearrange(patches,'bn (c ps1 ps2) r -> bn r ps1 ps2 c',ps1=ps,ps2=ps)
    patches_zc = patches.clone()
    patches_zc[...,ps//2,ps//2,:] = 0
    patches = rearrange(patches,'(b n) r ps1 ps2 c -> b n r (ps1 ps2 c)',n=N)
    patches_zc = rearrange(patches_zc,'(b n) r ps1 ps2 c -> b n r (ps1 ps2 c)',n=N)

    # -- contiguous for faiss --
    patches = patches.contiguous()
    patches_zc = patches_zc.contiguous()
    B,N,R,ND = patches.shape

    # -- faiss setup --
    res = faiss.StandardGpuResources()
    faiss_cfg = faiss.GpuIndexFlatConfig()
    faiss_cfg.useFloat16 = False
    faiss_cfg.device = cfg.gpuid

    # -- search each batch separately --
    sim_images = []
    for b in range(B):

        sim_frames = []
        for n in range(N):

            # -- setup faiss index --
            database = rearrange(patches[b],'n r l -> (n r) l')
            # database_zc = rearrange(patches_zc[b],'n r l -> (n r) l')
            database_zc = patches_zc[b,n]
            gpu_index = faiss.GpuIndexFlatL2(res, ND, faiss_cfg)
            gpu_index.add(database_zc)

            # -- run faiss search --
            query = patches_zc[b,n].contiguous()
            D, I = gpu_index.search(query,K+1)

            # -- locations of smallest K patches; no identity so no first column --
            S,V = D[:,1:], I[:,1:]

            # -- reshape to extract middle pixel from nearest patches --
            shape_str = 'hw k (ps1 ps2 c) -> hw k ps1 ps2 c'
            sim_patches = rearrange(database[V],shape_str,ps1=ps,ps2=ps)

            # -- create image from nearest pixels --
            sim_pixels = sim_patches[...,ps//2,ps//2,:]
            shape_str = '(h w) k c -> k (c h w)'
            sim_image = rearrange(sim_pixels,shape_str,h=H)

            # -- concat with original image --
            image_flat = rearrange(burst[n,b],'c h w -> 1 (c h w)')
            sim_image = torch.cat([image_flat,sim_image],dim=0)

            # -- shuffle across each k --
            if shuffle_k:
                index = torch.stack([torch.randperm(K+1) for _ in range(C*H*W)]).t().long()
                index = index.to(cfg.gpuid)
                sim_image.scatter_(0,index,sim_image)

            # -- add to frames --
            shape_str = 'k (c h w) -> 1 k c h w'
            sim_image = rearrange(sim_image,shape_str,h=H,w=W)
            sim_frames.append(sim_image)

            del query

        # -- concat on NumFrames dimension --
        sim_frames = torch.cat(sim_frames,dim=0)
        sim_images.append(sim_frames)
        del gpu_index

        # -- randomly shuffle pixels across k images --
        # sim_images[:,:] = sim_images[:,shuffle]

    # -- create batch dimension --
    sim_images = torch.stack(sim_images,dim=1)

    # -- visually inspect --
    vis = False
    if vis:
        print(burst.shape,sim_images.shape)
        sims = rearrange(sim_images[:,0],'n k c h w -> (n k) c h w')
        tv_utils.save_image(burst[:,0],'sim_search_tgt.png',nrow=N,normalize=True)
        tv_utils.save_image(sims,'sim_search_similar_images.png',nrow=K,normalize=True)
        # tv_utils.save_image(sim_images[0,0],'sim_search_similar_locations.png')


    # -- append to original images --
    burst_sim = sim_images
    # burst_sim = torch.cat([burst.unsqueeze(2),sim_images],dim=2)

    # -- compute differences to check --
    # for k in range(K+1):
    #     print(k,F.mse_loss(burst_sim[:,:,k],burst))
    # exit()

    return burst_sim


def create_k_grid(burst,G,shuffle=False,L=None):
    """
    :params burst: shape: [B,N,G,C,H,W]
    """

    # -- method v1 --
    k_ins,k_outs = [],[]
    for i in range(0,G,2):
        if i+1 >= G: break
        k_ins.append(i),k_outs.append(i+1)

    # -- shuffle order among pairs --
    L = len(k_ins)
    k_ins,k_outs = np.array(k_ins),np.array(k_outs)
    order = npr.permutation(L)
    k_ins,k_outs = k_ins[order],k_outs[order]
    swap = np.array([npr.permutation(2) for _ in range(L)])

    # -- shuffle order within pairs -- (just torch.scatter for numpy)
    kindex = np.c_[k_ins,k_outs]
    for l in range(L):
        k_ins[l],k_outs[l] = kindex[l,swap[l][0]],kindex[l,swap[l][1]]

    return k_ins,k_outs

def create_k_grid_v2(K,shuffle=False,L=None):
    x,y = np.arange(K),np.arange(K)
    xv,yv = np.meshgrid(x,y)
    k_ins,k_outs = xv.ravel(),yv.ravel()

    # -- only keep unique, non-eq pairs --
    pairs = []
    for kin,kout in zip(k_ins,k_outs):
        if kin >= kout: continue
        pairs.append([kin,kout])
    pairs = np.array(pairs)
    P = len(pairs)
    k_ins,k_outs = pairs[:,0],pairs[:,1]

    # -- shuffle if needed --
    if shuffle:
        ordering = npr.permutation(P)
        k_ins,k_outs = k_ins[ordering],k_outs[ordering]

    # -- limit number of combos --
    if not (L is None):
        k_ins,k_outs = k_ins[:L],k_outs[:L]

    return k_ins,k_outs


