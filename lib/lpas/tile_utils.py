# -- python imports --
from einops import rearrange

# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F


def aligned_burst_from_indices_global_dynamics(patches,t_indices,bl_indices):
    # new: 4:02, added n_indices
    R,B = patches.shape[:2]
    T = t_indices.shape[0]
    best_patches = []
    for b in range(B):
        patches_n = []
        for t in range(T):
            patches_n.append(patches[:,b,t_indices[t],bl_indices[b,t],:,:,:])
        patches_n = torch.stack(patches_n,dim=1)
        best_patches.append(patches_n)
    best_patches = torch.stack(best_patches,dim=1)
    return best_patches

def aligned_burst_image_from_indices_global_dynamics(patches,t_indices,block_indices):
    aligned_patches = aligned_burst_from_indices_global_dynamics(patches,t_indices,block_indices)
    R,PS = aligned_patches.shape[0],aligned_patches.shape[-1]
    H = int(np.sqrt(R))
    aligned = rearrange(aligned_patches[:,:,:,:,PS//2,PS//2],'(h w) b n c -> n b c h w',h=H)
    return aligned

def tile_burst_patches(burst,PS,T):

    # -- prepare --
    T,B,C,H,W = burst.shape
    dilation,padding,stride = 1,0,1
    unfold = nn.Unfold(PS,dilation,padding,stride)

    # -- all images at once --
    batch = rearrange(burst,'t b c h w -> (t b) c h w')

    # -- tile images with padding for unfold -- --
    tiled = tile_batch(batch,PS,T)

    # -- unfold to patches --
    tiled = rearrange(tiled,'tb n1 n2 c h w -> tb (n1 n2 c) h w')
    patches = unfold(tiled)

    # -- reshape patches for next use --
    patches = rearrange(patches,'tb (n c ps1 ps2) r -> tb r n ps1 ps2 c',c=C,ps1=PS,ps2=PS)
    patches = rearrange(patches,'(t b) r n ps1 ps2 c -> r b t n c ps1 ps2',t=T)

    return patches

def tile_batch(batch,PS,T):
    """
    Creates a tiled version of the input batch to be able to be rolled out using "unfold"
    
    The number of neighborhood pixels to include around each center pixel is "TH"
    The size of the patch around each chosen index (including neighbors) is "PS"
    
    We want to only pad the image once with "reflect". Padding an already padded image
    leads to a "reflection" of a "reflection", and this leads to even stranger boundary 
    condition behavior than whatever one "reflect" will do.

    We extend the image to its final size to apply "unfold" (hence the Hnew, Wnew)
    
    We tile the image w.r.t the neighborhood size so each original center pixel
    has TH neighboring patches.
    """
    B,C,H,W = batch.shape
    Hnew,Wnew = H + 2*(PS//2),W + 2*(PS//2)
    M = PS//2 + T//2 # reaching up T/2 center-pixels. Then reach up PS/2 more pixels
    batch_pad = F.pad(batch, [M,]*4, mode="reflect")
    tiled,idx = [],0
    for i in range(T):
        img_stack = []
        for j in range(T):
            img_stack.append(batch_pad[..., i:i + Hnew, j:j + Wnew])
        img_stack = torch.stack(img_stack, dim=1)
        tiled.append(img_stack)
    tiled = torch.stack(tiled,dim=1)
    return tiled

