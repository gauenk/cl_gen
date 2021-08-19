
# -- python imports --
import os,sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import matplotlib.pyplot as plt
from einops import rearrange,repeat
from pathlib import Path
from matplotlib.patches import ConnectionPatch
from scipy.ndimage.interpolation import affine_transform

# -- pytorch imports --
import torch

# -- project imports --
from pyutils import save_image
from patch_search.pixel.bootstrap_numba import fill_weights
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda

# -- local imports --
from configs import get_cfg_defaults


SAVE_DIR = Path("./figures/align/")
# gcolor = '#808080'
gcolor = '#000000'

def get_data_ij(data,i,j,nblocks,patchsize):
    ip = i# - nblocks//2
    jp = j# - nblocks//2
    si = ip
    ei = si+patchsize
    sj = jp
    ej = sj+patchsize
    return data[si:ei,sj:ej]

def create_gridspec(ngrids,patchsize,nframes,left=None,right=None,
                    wspace=None,hspace=None):
    # -- create subplots for blocks --
    hr = [1,patchsize,]*nframes# + [patchsize]
    gs_sub = GridSpec(1,ngrids,wspace=0.1,hspace=hspace,width_ratios=hr)
    gs_sub.update(left=left, right=right, wspace=wspace)
    ax = []
    for i in range(1):
        ax_i = []
        for j in range(ngrids):
            ax_ij = plt.subplot(gs_sub[i,j])
            ax_i.append(ax_ij)
        ax.append(ax_i)
    return ax

def create_paired_gridspec(nrows,ngrids,patchsize,nframes,left=None,right=None,
                           wspace=None,hspace=None):
    # -- create subplots for blocks --
    ax = []
    if nrows < 5:
        pad = 0.08
        fudge = 0.055
        step = 1./(nframes) - pad - fudge
        left,right = 0,step
    else:
        pad = 0.08
        fudge = 0.055
        step = 1./(nframes) - pad - fudge
        left,right = 0,step

    # -- init empty shape --
    ax = []
    for row in range(nrows):
        ax_row = []
        # for t in range(nframes):
        #     ax_row.append([])
        ax.append(ax_row)
        
    # -- create gridspec subplots --
    for t in range(nframes):
        hr = [1,patchsize]
        gs_sub = GridSpec(nrows,2,wspace=0.1,width_ratios=hr)
        gs_sub.update(left=left, right=right, wspace=wspace)

        # -- update left and right --
        print(left,right)
        left = right + pad
        right = left + step

        # -- append to weird list --
        for row in range(nrows):
            ax_ij = plt.subplot(gs_sub[row,0])
            ax[row].append(ax_ij)
            ax_ij = plt.subplot(gs_sub[row,1])
            ax[row].append(ax_ij)
    
    # -- final plot --
    delta = .97333 - .839999
    # left += pad*3./4
    right = left + delta + 0.018 # .97333 - .839999
    print(left,right)
    hr = [patchsize]
    gs_sub = GridSpec(nrows,1,wspace=0.1,width_ratios=hr)
    gs_sub.update(left=left, right=right, wspace=wspace)
    for row in range(nrows):
        ax_ij = plt.subplot(gs_sub[row,0])
        ax[row].append(ax_ij)

    # -- append to row --
    print(ax)
    return ax

def no_pointy_tics(ax):
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)

def boarder_axis(fig,ax,bcolor):
    bbox = ax.get_tightbbox(fig.canvas.renderer)
    x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
    # slightly increase the very tight bounds:
    xpad = 0.05 * width
    ypad = 0.05 * height
    fig.add_artist(plt.Rectangle((x0-0*xpad, y0-0*ypad),
                                 width+0*xpad, height+0*ypad,
                                 edgecolor=bcolor, linewidth=4, fill=False))

def get_burst_data(nblocks,patchsize,nframes,offset,ppf):

    # -- image --
    dyns = ppf*(nframes-1)
    zoom_size = 2*(nblocks//2) + patchsize + dyns
    image = np.random.rand(zoom_size, zoom_size) * 3
    print(image.shape)

    # -- image to burst --
    burst = []
    for t in range(nframes):
        ox,oy = offset[t]
        x = nblocks//2 - ox
        y = nblocks//2 - oy
        image_t = image[ox:ox+patchsize,oy:oy+patchsize]
        burst.append(image_t)
    burst = np.stack(burst)
    return burst

def plot_bootstrapping(noisy,clean):

    # -- settings --
    nblocks = 3
    patchsize = 5
    nframes = 3
    nsubsets = 8
    ppf = 1
    gpuid = 0

    # -- get image burst --
    offset = np.array([[1,0],[0,0],[0,1]])
    burst = get_burst_data(nblocks,patchsize,nframes,offset,ppf)
    burst /= burst.max()

    # -- get weights --
    device = f'cuda:{gpuid}'
    weights = torch.zeros((nsubsets,nframes),device=device)
    counts = torch.zeros((nsubsets,nframes),device=device)
    fill_weights(weights,counts,nsubsets,nframes,gpuid)
    weights = weights.cpu().numpy()
    weights += 1./nframes
    print(weights)

    # -- plot weights --
    postfix = "randmat"
    bcolor = '#00FFFF'
    create_weight_plot(None,None,weights,nblocks,
                       nframes,patchsize,postfix,bcolor)

    # -- plot weights @ image frames --
    postfix = "wsum"
    aves = create_weight_burst_plot(None,None,noisy,clean,burst,
                                    weights,nblocks,nframes,
                                    patchsize,postfix,bcolor)
    # -- plot average stack --
    postfix = "astack"
    create_ave_stack_plot(None,None,aves,postfix,bcolor)

def create_patch_plot_images(aves):

    ave_images = []
    nsubsets = aves.shape[0]
    for s in range(nsubsets):

        # -- init matplotlib --
        fig = Figure(figsize=(4,4),dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        # -- draw patch image --
        # ax.text(0.0,0.0,"Test", fontsize=45)
        frame = aves[s]
        ax.imshow(frame, cmap='Greys')
        ax.grid(which='major', axis='both',
                linestyle='-', color=gcolor, linewidth=3)
        ax.set_xticks(np.arange(-.5, frame.shape[1], 1));
        ax.set_yticks(np.arange(-.5, frame.shape[0], 1));
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.margins(0)
        no_pointy_tics(ax)
        canvas.draw()       # draw the canvas, cache the renderer
        fig.tight_layout(pad=0)
    
        # -- get image --
        # width, height = fig.get_size_inches() * fig.get_dpi()
        # width, height = int(width), int(height)
        # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        # image = image.reshape(height,width,3)
        canvas_b, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(canvas_b, np.uint8)
        image = image.reshape(height,width,4)
        image = image[:,:,:3]

        image_fl = image.astype(np.float) / 255.
        # print(image_fl)
        thimage = torch.FloatTensor(image_fl).type(torch.float)
        thimage = thimage.transpose(2,0)
        # print(thimage)
        # print(type(thimage))
        # print(thimage.shape)
        save_image(thimage,f"./image_{s}.png")
        
        ave_images.append(image)

        # -- close figure / reset matplotlib --
        plt.close("all")
        plt.clf()
    ave_images = np.stack(ave_images)
    return ave_images
        
def create_ave_stack_plot(input_fig,input_ax,aves,postfix,bcolor):

    # -- init --
    nsubsets = aves.shape[0]
    bg_val = -1

    # -- create patch plot images --
    aves = create_patch_plot_images(aves)
    nsubsets,h,w,color = aves.shape
    # aves = np.mean(aves[:,:,:,:],axis=-1)

    # -- stacked info --
    stacked_width = 2*w
    stacked_height = int( (h + (nsubsets - 1)*h/2))
    stacked = np.full((stacked_height,stacked_width,color), bg_val)
    
    # -- affine mat --
    # T = np.array([[1,-1],[0,1]]) # shear parallel "x"
    T = np.array([[1,0],[-1,1]]) # shear parallel "y"

    # -- create stack --
    print(stacked.shape)
    for s in range(nsubsets):
        a_sub = nsubsets - s - 1
        o = (nsubsets-s-1)*h/2# * 2./3
        print(o)
        for c in range(color):
            stacked_c = stacked[:,:,c]
            out = affine_transform(aves[a_sub,:,:,c].astype(np.float),
                                   T,
                                   offset=[-o,o],
                                   output_shape=stacked_c.shape,
                                   mode='constant',
                                   cval=bg_val)
            stacked_c[out != bg_val] = out[out != bg_val]

    # -- create subplots --
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    fig.canvas.draw()
    ax.axis("off")

    # -- create figure --
    vmin,vmax = 0,1.
    # stacked = np.ma.masked_where(np.isclose(stacked,-1.),stacked)
    # stacked[stacked < -0.9] = np.nan
    stacked_alpha = np.zeros((stacked_height,stacked_width,4))
    stacked_alpha[:,:,:3] = stacked
    stacked_alpha[:,:,3] = stacked.max() * (stacked[:,:,0] > -0.9)
    stacked_alpha = stacked_alpha.astype(np.int)
    ax.imshow(stacked_alpha, cmap='Greys',interpolation='none')#,vmin=vmin,vmax=1.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    no_pointy_tics(ax)

    # -- save figure --
    fname = SAVE_DIR / f"bootstrapping_{postfix}.png"
    plt.savefig(fname,transparent=True,bbox_inches='tight',dpi=300)
    plt.close("all")
    plt.clf()

def create_weight_burst_plot(input_fig,input_ax,noisy,clean,burst,weights,
                             nblocks,nframes,patchsize,postfix,bcolor):

    # -- create colormap --
    weights = np.copy(weights)
    weights_nmlz = weights - weights.min()
    cmap = 'Greys'

    # -- create subplots --
    fig = plt.figure(figsize=(6,10))
    fig.canvas.draw()

    # if input_ax is None:
    #     height,width = 8,8
    #     fig,ax = plt.subplots(figsize=(height,width))
    # else: fig,ax = input_fig,input_ax

    # -- create gridspec --
    nrows = weights.shape[0]
    ncols = 2*nframes + 1
    axes = create_paired_gridspec(nrows,ncols,patchsize,nframes,
                                  left=None,right=None,wspace=None,hspace=None)
    
    # -- create row --
    aves = []
    vmin,vmax = -1./nframes,1.
    for row in range(nrows):
        ave = 0
        weight_row = weights[row,:]
        weight_row_nmlz = weights_nmlz[row,:]
        print(weight_row)
        for t in range(nframes):
            frame = burst[t][:patchsize,:patchsize]
            weight_nmlz = weight_row_nmlz[t].reshape(1,1,1)
            weight = weight_row[t].reshape(1,1,1)

            #ave += weight_row[t] * frame
            ave += weight_row[t] * noisy[t].cpu()

            w_axis = axes[row][2*t]
            w_axis.imshow(weight, cmap='Greys',vmin=vmin,vmax=1.0)
            w_axis.set_xticklabels([])
            w_axis.set_yticklabels([])
            w_axis.set_aspect('equal')
            no_pointy_tics(w_axis)
    
            frame_ax = axes[row][2*t+1]

            # -- replace weight with image --
            weight = noisy[t] - noisy[t].min()
            weight /= weight.max()
            weight = weight.cpu().numpy()
            frame_ax.imshow(weight, vmin=0,vmax=1.0)
            # -- use weight --
            # frame_ax.imshow(frame, cmap='Greys',vmin=vmin,vmax=1.0)
            # frame_ax.grid(which='major', axis='both',
            #         linestyle='-', color=gcolor, linewidth=2)
            # frame_ax.set_xticks(np.arange(-.5, frame.shape[1], 1));
            # frame_ax.set_yticks(np.arange(-.5, frame.shape[0], 1));
            frame_ax.set_xticklabels([])
            frame_ax.set_yticklabels([])
            frame_ax.set_aspect('equal')
            no_pointy_tics(frame_ax)
    
        # -- create averaged patch in row --
        aves.append(ave)
        frame_ax = axes[row][-1]
        frame_ax.imshow(ave, cmap='Greys')
        # frame_ax.imshow(ave, cmap='Greys')
        # frame_ax.grid(which='major', axis='both',
        #         linestyle='-', color=gcolor, linewidth=2)
        # frame_ax.set_xticks(np.arange(-.5, frame.shape[1], 1));
        # frame_ax.set_yticks(np.arange(-.5, frame.shape[0], 1));
        frame_ax.set_xticklabels([])
        frame_ax.set_yticklabels([])
        frame_ax.set_aspect('equal')
        no_pointy_tics(frame_ax)

        # -- write plus signs --
        method = "use_image"
        if method == "use_pix":
            if nrows < 5:
                left,top = 5.75,2.25
            else:
                left,top = 5.25,2.25
        else:
            left,top = 70.75,35.25
        for t in range(nframes-1):
            axes[row][2*t+1].text(left, top, r'$+$', fontsize=20)
        axes[row][2*nframes-1].text(left, top, r'$=$', fontsize=20)                
        


    # -- save frame - ave image --
    ave = torch.stack(aves,dim=0)
    tosave = rearrange(torch.mean(noisy.cpu(),dim=0) - ave,'s h w c -> s c h w')
    save_image(tosave,SAVE_DIR/"./bootstrap_noisy_delta_ave.png")

    # -- save figure --
    fname = SAVE_DIR / f"bootstrapping_{postfix}.png"
    plt.savefig(fname,transparent=True,bbox_inches='tight',dpi=300)
    plt.close("all")
    plt.clf()

    # -- stack and return aves --
    aves = np.stack(aves)
    return aves


def create_weight_plot(input_fig,input_ax,weights,
                       nblocks,nframes,patchsize,postfix,bcolor):

    # -- create colormap --
    # weights -= weights.min()
    cmap = 'Greys'

    # -- create subplots --
    if input_ax is None:
        height,width = 8,8
        fig,ax = plt.subplots(figsize=(height,width))
    else: fig,ax = input_fig,input_ax
    
    # -- fill subplots for blocks --
    vmin,vmax = -1./nframes,1.
    ax.imshow(weights, cmap='Greys',vmin=vmin,vmax=vmax)
    # ax.imshow(weights, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both',
         linestyle='-', color=gcolor, linewidth=2)
    ax.set_xticks(np.arange(-.5, weights.shape[1], 1));
    ax.set_yticks(np.arange(-.5, weights.shape[0], 1));
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    no_pointy_tics(ax)

    if input_ax is None:
        fname = SAVE_DIR / f"bootstrapping_{postfix}.png"
        plt.savefig(fname,transparent=True,bbox_inches='tight',dpi=300)
        plt.close("all")
        plt.clf()
    
def main():
    seed = 234
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -- settings --
    cfg = get_cfg_defaults()
    cfg.use_anscombe = True
    cfg.noise_params.ntype = 'g'
    cfg.noise_params.g.std = 25.
    cfg.nframes = 3
    cfg.num_workers = 0
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.dynamic_info.ppf = 10
    cfg.nblocks = 3
    cfg.patchsize = 10
    cfg.gpuid = 1
    cfg.device = f"cuda:{cfg.gpuid}"

    # -- load dataset --
    data,loaders = load_image_dataset(cfg)
    train_iter = iter(loaders.tr)

    # -- fetch sample --
    sample = next(train_iter)
    sample_to_cuda(sample)

    # -- unpack data --
    noisy,clean = sample['noisy'],sample['burst']

    # -- save ave image --
    save_image(torch.mean(noisy[:,0],dim=0),SAVE_DIR/"./bootstrap_noisy_ave.png")

    # -- format for plots --
    print("noisy.shape",noisy.shape)
    noisy = rearrange(noisy[:,0],'t c h w -> t h w c')
    clean = rearrange(clean[:,0],'t c h w -> t h w c')

    plot_bootstrapping(noisy,clean)


if __name__ == "__main__":
    main()
