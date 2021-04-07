"""
Code to search for the aligned patches from bursts

"""

# -- python imports --
import numpy as np
from einops import rearrange

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

# -- project imports --
from pyutils.misc import images_to_psnrs

def abp_search(cfg,burst):
    """
    burst: shape = (N,B,C,H,W)
    """
    N,B,C,H,W = burst.shape
    n_grid = torch.arange(N)
    mid = torch.LongTensor([N//2])
    no_mid = torch.LongTensor(np.r_[np.r_[:N//2],np.r_[N//2+1:N]])

    PS = cfg.patchsize
    NH = cfg.nh_size
    patches = tile_burst_patches(burst,PS,NH)
    gn_noisy = torch.normal(patches,75./255.)
    alpha = 4.
    pn_noisy = torch.poisson(alpha* (patches+0.5) )/alpha
    # patches.shape = (R,B,N,NH^2,C,PS,PS)

    
    unrolled_img = tile_batches_to_image(gn_noisy,NH,PS,H)
    tv_utils.save_image(unrolled_img,"gn_noisy.png",normalize=True,range=(-.5,.5))

    unrolled_img = tile_batches_to_image(pn_noisy,NH,PS,H)
    tv_utils.save_image(unrolled_img,"pn_noisy.png",normalize=True,range=(0.,1.))


    """
    For each N we pick 1 element from NH^2.

    Integer programming problem with state space size (NH^2)^(N-1)
    
    Dynamic programming to the rescue. Divide-and-conquer?
    """
    if N != 5: print("\n\n\n\n\n[WARNING]: This program is only coded for N == 5\n\n\n\n\n")

    best_idx = torch.zeros(B,N).type(torch.long)
    # features = gn_noisy
    features = pn_noisy
    for b in range(B):
        features_b = features[:,[b]]
        REF_NH = NH**2//2 + NH//2
        mid = torch.LongTensor([REF_NH])
        # m_patch = features_b[:,:,N//2,REF_NH,:,:,:]
        # s_features = features[:,:,no_mid,select,:,:,:] # (R,B,N-1,C,PS,PS)
        # ave_features = torch.mean(s_features,dim=2) # (R,B,C,PS,PS)
        # psnr = F.mse_loss(ave_features,m_patch).item()
    
        left_idx = torch.LongTensor(np.r_[:N//2])
        # best_left = torch.zeros(len(left_idx))
        best_left = abp_search_split(features_b,left_idx,PS,NH)
    
        right_idx = torch.LongTensor(np.r_[N//2+1:N])
        # best_right = torch.zeros(len(right_idx))
        best_right = abp_search_split(features_b,right_idx,PS,NH)
        
        best_idx_b = torch.cat([best_left,mid,best_right],dim=0).type(torch.long)
        best_idx[b] = best_idx_b

    best_idx = best_idx.to(patches.device)
    best_patches = []
    for b in range(B):
        patches_n = []
        for n in range(N):
            patches_n.append(patches[:,b,n,best_idx[b,n],:,:,:])
        patches_n = torch.stack(patches_n,dim=1)
        best_patches.append(patches_n)
    best_patches = torch.stack(best_patches,dim=1)
    # best_patches = torch.stack([patches[:,:,n,best_idx[b,n],:,:,:] for n in range(N)],dim=1)
    print("bp",best_patches.shape)
    ave_patches = torch.mean(best_patches,dim=2)
    set_imgs = rearrange(best_patches[:,:,:,:,PS//2,PS//2],'(h w) b n c -> b n c h w',h=H)
    ave_img = rearrange(ave_patches[:,:,:,PS//2,PS//2],'(h w) b c -> b c h w',h=H)
    print("A",ave_img.shape)
    return best_idx,ave_img,set_imgs

def abp_search_split(patches,burst_idx,PS,NH):
    R,B,N = patches.shape[:3]
    FMAX = np.finfo(np.float).max
    REF_NH = NH**2//2 + NH//2
    ref_patch = patches[:,:,N//2,REF_NH,:,:,:]
    assert len(burst_idx) == 2, "Chunks of 2 right now."
    # best_value,best_select = FMAX*torch.ones(R,B),torch.zeros(R,B,2).type(torch.long)
    best_value,best_select = FMAX,None
    for i in range(NH**2):
        for j in range(NH**2):
            i_patch = patches[:,:,burst_idx[0],i,:,:,:] 
            j_patch = patches[:,:,burst_idx[1],j,:,:,:] 
            select = torch.LongTensor([i,j])
            ave_patches = torch.mean(patches[:,:,burst_idx,select,:,:,:],dim=2)

            value,count = 0,0
            compare_patches = [i_patch,j_patch,ave_patches,ref_patch]
            for k,k_patch in enumerate(compare_patches):
                for l,l_patch in enumerate(compare_patches):
                    if k >= l: continue
                    value += F.mse_loss(k_patch,l_patch).item()
                    # k_mean = k_patch.mean().item()
                    # l_mean = l_patch.mean().item()
                    # var = (75./255.)**2
                    # value += (F.mse_loss(k_patch-k_mean,l_patch-l_mean).item() - 2*var)**2
                    count += 1
            value /= count
            if value < best_value:
                best_value = value
                best_select = select
    print(best_select)
    return best_select

def tile_burst_patches(burst,PS,NH):
    N,B,C,H,W = burst.shape
    dilation,padding,stride = 1,0,1
    unfold = nn.Unfold(PS,dilation,padding,stride)
    batch = rearrange(burst,'n b c h w -> (n b) c h w')
    tiled = tile_batch(batch,PS,NH)

    save_ex = rearrange(tiled,'nb t1 t2 c h w -> nb t1 t2 c h w')
    save_ex = save_ex[:,NH//2,NH//2,:,:,:]
    tv_utils.save_image(save_ex,"save_ex.png",normalize=True)
    print("tiled_ex.shape",save_ex.shape)

    tiled = rearrange(tiled,'nb t1 t2 c h w -> nb (t1 t2 c) h w')
    patches = unfold(tiled)
    patches = rearrange(patches,'nb (t c ps1 ps2) r -> nb r t ps1 ps2 c',c=C,ps1=PS,ps2=PS)
    patches = rearrange(patches,'(n b) r t ps1 ps2 c -> r b n t c ps1 ps2',n=N)

    unrolled_img = tile_batches_to_image(patches,NH,PS,H)
    tv_utils.save_image(unrolled_img,"unroll_img.png",normalize=True)

    return patches
    
def tile_batches_to_image(patches,NH,PS,H):
    REF_NH = NH**2//2 + NH//2
    unrolled_img = patches[:,:,:,REF_NH,:,PS//2,PS//2]
    unrolled_img = rearrange(unrolled_img,'(h w) b n c -> (b n) c h w',h=H)
    return unrolled_img

def tile_batch(batch,PS,NH):
    """
    Creates a tiled version of the input batch to be able to be rolled out using "unfold"
    
    The number of neighborhood pixels to include around each center pixel is "NH"
    The size of the patch around each chosen index (including neighbors) is "PS"
    
    We want to only pad the image once with "reflect". Padding an already padded image
    leads to a "reflection" of a "reflection", and this leads to even stranger boundary 
    condition behavior than whatever one "reflect" will do.

    We extend the image to its final size to apply "unfold" (hence the Hnew, Wnew)
    
    We tile the image w.r.t the neighborhood size so each original center pixel
    has NH neighboring patches.
    """
    B,C,H,W = batch.shape
    Hnew,Wnew = H + 2*(PS//2),W + 2*(PS//2)
    M = PS//2 + NH//2 # reaching up NH/2 center-pixels. Then reach up PS/2 more pixels
    batch_pad = F.pad(batch, [M,]*4, mode="reflect")
    tiled,idx = [],0
    for i in range(NH):
        img_stack = []
        for j in range(NH):
            img_stack.append(batch_pad[..., i:i + Hnew, j:j + Wnew])
            # -- test we are correctly cropping --
            # print(i+Hnew,j+Wnew,H,W,batch_pad.shape)
            # cmpr = tvF.crop(img_stack[j],PS//2,PS//2,H,W)
            # print("i,j,idx",i,j,idx,images_to_psnrs(batch,cmpr))
            # idx += 1
        img_stack = torch.stack(img_stack, dim=1)
        tiled.append(img_stack)
    tiled = torch.stack(tiled,dim=1)
    return tiled

