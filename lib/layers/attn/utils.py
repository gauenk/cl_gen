
# -- python imports --
import pdb
from einops import rearrange
import matplotlib.pyplot as plt
from pathlib import Path

# -- pytorch imports --
import torch
import torchvision.utils as tv_utils

# -- project imports --
import settings

def squared_euclidean_distance(a, b):
    b = b.T
    a2 = torch.sum(torch.pow(a,2),1,keepdims=True)
    b2 = torch.sum(torch.pow(b,2),0,keepdims=True)
    ab = torch.matmul(a, b)
    d = a2 - 2*ab + b2
    return d

def color_quantize(pic, np_clusters, d_model):
    if len(pic.shape) == 5:
        return color_quantize_5dim(pic, np_clusters, d_model)
    elif len(pic.shape) == 4:
        return color_quantize_4dim(pic, np_clusters, d_model)

def color_quantize_5dim(pic, np_clusters, d_model):
    N,BS,C,H,W = pic.shape
    clusters = torch.FloatTensor(np_clusters).to(pic.device)
    pic = rearrange(pic,'n bs c h w -> (n bs h w) c')
    d = squared_euclidean_distance(pic,clusters)

    # -- extract clostest cluster index value for each pixel --
    d_argmin = torch.min(d, 1)[1]
    d_argmin = rearrange(d_argmin,'(n bs 1 h w) -> n bs 1 h w',n=N,bs=BS,h=H,w=W)

    # -- expand each index into indicator var --
    one_hot = torch.FloatTensor(N,BS,d_model,H,W).zero_().to(d_argmin.device)
    one_hot = one_hot.scatter_(2,d_argmin,1)

    return one_hot

def color_quantize_4dim(pic, np_clusters, d_model):
    BS,C,H,W = pic.shape
    clusters = torch.FloatTensor(np_clusters).to(pic.device)
    pic = rearrange(pic,'bs c h w -> (bs h w) c')
    d = squared_euclidean_distance(pic,clusters)

    # -- extract clostest cluster index value for each pixel --
    d_argmin = torch.min(d, 1)[1]
    d_argmin = rearrange(d_argmin,'(bs 1 h w) -> bs 1 h w',bs=BS,h=H,w=W)

    # -- expand each index into indicator var --
    one_hot = torch.FloatTensor(BS,d_model,H,W).zero_().to(d_argmin.device)
    one_hot = one_hot.scatter_(2,d_argmin,1)

    return one_hot

def color_dequantize(pic_logits, np_clusters):
    BS,D,H,W = pic_logits.shape
    clusters = torch.FloatTensor(np_clusters).to(pic_logits.device)

    # -- raster -- 
    pic_logits = rearrange(pic_logits,'bs c h w -> (bs h w) c')
    # -- index clusters --
    pic = rearrange(clusters[pic_logits.argmax(1)],'(bs h w) c -> bs c h w',bs=BS,c=3,h=H,w=W)
    # -- conv to pixel space --
    pic = 127.5 * (pic + 1.0)
    return pic
    

def plot_patches(patches,N,i_p):
    # in_patches = rearrange(in_patches,'(b n) (c i) r -> r (n i) b c',b=BS,i=i_p**2)
    # patches: repeats, number of frames x total num of pixels, batch size, num of channels
    patches = patches.cpu()
    R,NI,BS,C = patches.shape
    I = N // NI
    for r in range(R):
        images_th = rearrange(patches[r],'(n i_h i_w) bs c -> (bs n) c i_h i_w',n=N,i_h=i_p)
        plot_image = tv_utils.make_grid(images_th,nrow=BS,normalize=True)
        plot_image = rearrange(plot_image,'c h w -> h w c')
        plt.close("all")
        fig,ax = plt.subplots(figsize=(8,8))
        ax.imshow(plot_image)
        path = Path(settings.ROOT_PATH) / Path("./output/attn/layerdef_patches_attn_{}.png".format(r))
        plt.savefig(path,dpi=300)
    plt.close("all")
    
