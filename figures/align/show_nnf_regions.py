
# -- setup paths --
import sys,os
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/experiments/noisy_burst_pixel_alignment/")

# -- python imports --
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- torch imports --
import torch
import torch.nn.functional as F

# -- python plotting imports --
import matplotlib
matplotlib.use("agg")
matplotlib.rcParams['text.usetex'] = True
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# -- project imports --
from pyplots.legend import add_legend
from pyutils import print_tensor_stats
from align.xforms import flow_to_pix,align_from_flow
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda
from unsup_denoising.experiments.compare_to_competitors._aligned_methods import get_align_method

# -- local imports --
from configs import get_cfg_defaults

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def vprint(*args,**kwargs):
    verbose = True
    if verbose:
        print(*args,**kwargs)

def format_axix(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)

def draw_box_on_ax(ax,x,y,ps,color):
    t,l = y-ps//2,x-ps//2
    rect = patches.Rectangle((t,l),ps,ps,
                             linewidth=1,
                             edgecolor=color,
                             facecolor=color,
                             alpha = 0.25
    )
    ax.add_patch(rect)

def compute_alignment_quality(clean,ref_pix,box_est,ps,t):

    # -- get ref patch --
    nframes = clean.shape[0]
    x,y = ref_pix.x,ref_pix.y
    top,left = y-ps//2,x-ps//2
    ref_patch = clean[nframes//2][top:top+ps,left:left+ps]

    # -- pad image --
    pad = (ps//2,)*4
    clean_t = clean[[t]].transpose(3,1)
    clean_t = F.pad(clean_t,pad,mode='reflect')[0].transpose(2,0)

    # -- get patch --
    x,y = box_est[0],box_est[1]
    x,y = x+pad[0],y+pad[0]
    top,left = y-ps//2,x-ps//2
    prop_patch = clean_t[top:top+ps,left:left+ps]

    # -- compute diff --
    diff = torch.sum((ref_patch - prop_patch)**2).item()

    return diff

def plot_boxes(noisy,clean,aligned,field,boxes,ref_pix,ps,seed):


    # -- plot settings --
    colors = {'global_l2':'yellow','local_l2':'purple','local_bs':'blue','gt':'red'}

    # -- shapes --
    nframes,nimages,H,W,ncolors = noisy.shape
    nframes,nimages,H,W,ncolors = clean.shape
    nframes_m1,nimages,bH,bW,two = boxes['gt'].shape
    ref_t,iindex = nframes//2,0

    # -- only one image --
    noisy = noisy[:,iindex]
    clean = clean[:,iindex]

    # -- create figs --
    fig,ax = plt.subplots(1,nframes+1,figsize=(8,4))

    # -- format images on axis --
    for t in range(nframes+1):
        format_axix(ax[t])

    # -- draw images on axis --
    for t in range(nframes):
        ax[t].imshow(clean[t])
        # ax[t].imshow(noisy[t])


    # -- init xlabel --
    xlabels = []
    for t in range(nframes): xlabels.append("")

    # -- draw boxes on axis --
    fields = ['local_bs','local_l2','global_l2','gt']
    for field in fields:
        for t in range(nframes):
            if t == ref_t: # plot reference patch
                color = "blue"
                draw_box_on_ax(ax[t],ref_pix.x,ref_pix.y,ps,color)
                xlabels[t] = "Reference"
            else:
                box_t = t-1 if t > ref_t else t # correct for no ref_frame boxes
    
                # -- est. box with noisy images using "field" method --
                box_xy = boxes[field][box_t,iindex][ref_pix.x,ref_pix.y]
                color = colors[field]
                draw_box_on_ax(ax[t],box_xy[0],box_xy[1],ps,color)
                quality = compute_alignment_quality(clean,ref_pix,box_xy,ps,t)
    
                # -- gt box with clean images --
                # box_xy = boxes['gt'][box_t,iindex][ref_pix.x,ref_pix.y]
                # color = colors['gt']
                # draw_box_on_ax(ax[t],box_xy[0],box_xy[1],ps,color)
                # quality_gt = compute_alignment_quality(clean,ref_pix,box_xy,ps,t)
    
                # -- title using quality --
                sfield = field.replace("_"," ").title()
                xlabels[t] += "[%s] %2.1f\n" % (sfield,quality)

    # -- write xlabel --
    for t in range(nframes):
        if t == ref_t:
            ax[t].set_xlabel(xlabels[t],fontsize=12,loc='center')
        else:
            xlabels[t] = xlabels[t][:-1] # rm last newline
            ax[t].set_xlabel(xlabels[t],fontsize=12,loc='left')

    # -- add legend --
    titles = []
    handles = []
    for field in fields:
        sfield = field.replace("_"," ").title()
        titles.append(sfield)
        pop = patches.Patch(color=colors[field], label=sfield)
        handles.append(pop)
    ax[-1].axis("off")
    box = ax[-2].get_position()
    ax[-1].set_position([box.x0, box.y0,
                         box.width, box.height])
    add_legend(ax[-1],"Methods",titles,handles,shrink = False,fontsize=12,
               framealpha=0.0,ncol=1)

    # -- save plot --
    plt.savefig(f"./show_nnf_regions_{seed}.png",
                transparent=True,dpi=300,bbox_inches='tight')
    plt.close("all")
    plt.clf()

def boxes_from_flow(flow,h,w):
    isize = edict({'h':h,'w':w})
    flow_rs = rearrange(flow,'b (h w) t two -> t b h w two',h=h,w=w)
    pix = flow_to_pix(flow,isize=isize)
    pix = rearrange(pix,'b (h w) t two -> t b h w two',h=h,w=w)
    return pix

def run_with_seed(seed):
    
    # -- settings --
    cfg = get_cfg_defaults()
    cfg.use_anscombe = False
    cfg.noise_params.ntype = 'g'
    cfg.noise_params.g.std = 10.
    cfg.nframes = 5
    cfg.patchsize = 11

    # -- seeds --
    cfg.seed = seed
    # cfg.seed = 123 # sky of a forest 
    # cfg.seed = 345 # handrail and stairs
    # cfg.seed = 567 # cloudy blue sky
    # cfg.seed = 567 # cloudy blue sky

    # -- set seed --
    set_seed(cfg.seed)

    # -- load dataset --
    data,loaders = load_image_dataset(cfg)
    train_iter = iter(loaders.tr)

    # -- fetch sample --
    sample = next(train_iter)
    sample_to_cuda(sample)

    # -- unpack data --
    noisy,clean = sample['noisy'],sample['burst']
    nframes,nimages,ncolors,H,W = noisy.shape
    isize = edict({'h':H,'w':W})
    
    # -- boxes for plotting --
    boxes = edict()
    aligned = edict()
    
    # -- compute clean nnf --
    vprint("[start] clean nnf.")
    align_fxn = get_align_method(cfg,"l2_global")
    aligned.gt,flow = align_fxn(clean,None,None)
    boxes.gt = boxes_from_flow(flow,H,W)
    vprint("[done] clean nnf.")

    # -- compute nnf --
    vprint("[start] global nnf.")
    align_fxn = get_align_method(cfg,"l2_global")
    _,flow = align_fxn(noisy,None,None)
    aligned.global_l2 = align_from_flow(clean,flow,cfg.nblocks,isize=isize)
    boxes.global_l2 = boxes_from_flow(flow,H,W)
    vprint("[done] global nnf.")

    # -- compute local nnf --
    vprint("[start] local nnf.")
    align_fxn = get_align_method(cfg,"l2_local")
    _,flow = align_fxn(noisy,None,None)
    aligned.local_l2 = align_from_flow(clean,flow,cfg.nblocks,isize=isize)
    boxes.local_l2 = boxes_from_flow(flow,H,W)
    vprint("[done] local nnf.")

    # -- compute proposed score --
    vprint("[start] bootstrapping.")
    align_fxn = get_align_method(cfg,"bs_local_v2")
    _,flow = align_fxn(noisy,None,None)
    aligned.local_bs = align_from_flow(clean,flow,cfg.nblocks,isize=isize)
    boxes.local_bs = boxes_from_flow(flow,H,W)
    vprint("[done] bootstrapping.")

    # -- reshape to image --
    noisy = rearrange(noisy,'t b c h w -> t b h w c')
    clean = rearrange(clean,'t b c h w -> t b h w c')

    # -- normalize to [0,1] --
    noisy -= noisy.min()
    clean -= clean.min()
    noisy /= noisy.max()
    clean /= clean.max()

    # -- clamp to [0,1] --
    # noisy = noisy.clamp(0,1)
    # clean = clean.clamp(0,1)

    # print_tensor_stats("noisy",noisy)
    # print_tensor_stats("clean",clean)

    # -- cuda to cpu --
    noisy = noisy.cpu()
    clean = clean.cpu()
    for field in boxes.keys():
        boxes[field] = boxes[field].cpu().numpy()

    # -- plot boxes for middle pix --
    ref_pix = edict({'x':H//2,'y':W//2})
    field = 'global_l2'
    plot_boxes(noisy,clean,aligned,field,boxes,ref_pix,cfg.patchsize,seed)

def main():
    pid = os.getpid()
    print(f"PID: {pid}")
    nrands = 20
    for r in range(nrands):
        random_seed = int(torch.rand(1)*100)
        print(f"random_seed: [{random_seed}]")
        run_with_seed(random_seed)

if __name__ == "__main__":
    main()
