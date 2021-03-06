

# --  python imports --
import numpy as np
import numpy.random as npr
from einops import rearrange,repeat
from easydict import EasyDict as edict

# --  pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils import apply_sobel_filter,global_flow_to_blocks,images_to_psnrs,align_burst_from_block

# -- [local] project imports --
from .utils import get_ref_block_index

def divide_frame_size(fs,num):
    return fs // num + 1

def create_image_tiles(clean_vol,noisy_vol,flow,min_frame_size):

    # -- unpack --
    T,H2,B,P,C,PS,PS = clean_vol.shape

    # -- get correct tiles --
    flow = flow.cpu()
    nblocks = int(np.sqrt(H2))
    ref_blocks = global_flow_to_blocks(flow,nblocks).T

    # -- reference blocks --
    REF_H = get_ref_block_index(nblocks)
    clean = clean_vol[:,REF_H]
    noisy = noisy_vol[:,REF_H]

    # # -- removes dynamics: index with for loops --
    # device = clean_vol.device
    # ref_blocks = ref_blocks.to(device)
    # clean = torch.zeros((T,B,P,C,PS,PS),device=device)
    # noisy = torch.zeros((T,B,P,C,PS,PS),device=device)
    # for i in range(ref_blocks.shape[0]):
    #     for j in range(ref_blocks.shape[1]):
    #         clean[i,j] = clean_vol[i,ref_blocks[i,j],j]
    #         noisy[i,j] = noisy_vol[i,ref_blocks[i,j],j]
        
    # -- move patches across width --
    clean = rearrange(clean,'t b p c h w -> t b c h (w p)')
    noisy = rearrange(noisy,'t b p c h w -> t b c h (w p)')

    # -- tile images --
    rh = divide_frame_size(min_frame_size,clean.shape[-2])
    rw = divide_frame_size(min_frame_size,clean.shape[-1])
    shape_str = 't b c ph pw -> t b c (rh ph) (rw pw)'
    clean_tile = repeat(clean,shape_str,rh=rh,rw=rw)
    noisy_tile = repeat(noisy,shape_str,rh=rh,rw=rw)

    # -- crop to frame size --
    fs = min_frame_size
    clean_tile = tvF.crop(clean_tile,0,0,fs,fs)
    noisy_tile = tvF.crop(noisy_tile,0,0,fs,fs)
    tiles = edict({'clean':clean_tile,'noisy':noisy_tile})

    # align_burst_from_block(clean_tile,ref_block,nblocks,"global")

    return tiles

def create_image_volumes(cfg,clean,noisy):
    npatches,nblocks,patchsize = cfg.npatches,cfg.nblocks,cfg.patchsize
    ref_noisy = torch.mean(noisy,dim=0)
    patch_tl = sample_patch_locations(ref_noisy,npatches,patchsize,nblocks)
    clean_vol,noisy_vol = [],[]
    for b,tl_b in enumerate(patch_tl):
        clean_vol.append(crop_burst_to_blocks(clean[:,b],nblocks,tl_b,patchsize))
        noisy_vol.append(crop_burst_to_blocks(noisy[:,b],nblocks,tl_b,patchsize))
    clean_vol = torch.stack(clean_vol)
    noisy_vol = torch.stack(noisy_vol)

    # -- block grid 1st --
    clean_vol = rearrange(clean_vol,'b t h2 p c ps1 ps2 -> t h2 b p c ps1 ps2')
    noisy_vol = rearrange(noisy_vol,'b t h2 p c ps1 ps2 -> t h2 b p c ps1 ps2')

    vols = edict({'clean':clean_vol,'noisy':noisy_vol})
    return vols

def sample_patch_locations(full_image,P,patchsize,nblocks):
    # -- shape to include batch --
    if len(full_image.shape) == 3:
        full_image = full_image.unsqueeze(0)
        
    # -- get edges --        
    B,C,H,W = full_image.shape
    edges = apply_sobel_filter(full_image)
    B,H,W = edges.shape
    S = 3*P

    # -- sample based on edges and patchsize --
    buf,eps = 3,1e-13
    fs = patchsize//2 + buf + nblocks//2
    inner = edges[:,fs:-fs,fs:-fs]
    topP = torch.topk(inner.reshape(B,-1),S,largest=True).values
    init_tl_list = []
    for b in range(B):
        init_tl_b = []
        for s in range(S):
            maxs = topP[b,s]
            indices = torch.nonzero( torch.abs(inner[b] - maxs) < eps)
            for i in range(indices.shape[0]):
                init_tl = [int(x)+fs-patchsize//2 for x in indices[i]]
                init_tl_b.append(init_tl)
        order = npr.permutation(len(init_tl_b))[:P]
        init_tl_b = [init_tl_b[x] for x in order]
        init_tl_list.append(init_tl_b)
    init_tl_list = torch.LongTensor(init_tl_list)
    return init_tl_list

def crop_burst_to_blocks(full_burst,nblocks,init_tl_list,patchsize):
    P,H,ps = len(init_tl_list),nblocks,patchsize
    blocks = []
    for p in range(P):
        t,l = init_tl_list[p]
        blocks_p = []
        for i in range(-H//2+1,H//2+1):
            for j in range(-H//2+1,H//2+1):
                crop = tvF.crop(full_burst,t+i,l+j,ps,ps)
                # if i == 0 and j == 0: save_image(crop,f"crop_{p}.png")
                blocks_p.append(crop)
        blocks_p = torch.stack(blocks_p,dim=0)
        blocks.append(blocks_p)
    blocks = torch.stack(blocks,dim=0)
    blocks = rearrange(blocks,'p h2 t c h w -> t h2 p c h w',p=P)

    # T,H2,P,C,H,W = blocks.shape
    # REF_T = T//2
    # REF_H = get_ref_block_index(nblocks)
    # for t in range(T):
    #     for h2 in range(H2):
    #         psnrs = images_to_psnrs(blocks[t,h2],blocks[REF_T,REF_H])
    #         print(t,h2,np.mean(psnrs))

    return blocks

