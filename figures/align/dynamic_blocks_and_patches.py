"""


"""

# -- python imports --
import os,sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import ConnectionPatch


# -- pytorch imports --
import torch

# -- project imports --

SAVE_DIR = Path("./figures/align/")

def get_data_ij(data,i,j,nblocks,patchsize):
    ip = i# - nblocks//2
    jp = j# - nblocks//2
    si = ip
    ei = si+patchsize
    sj = jp
    ej = sj+patchsize
    return data[si:ei,sj:ej]

def create_gridspec(nblocks,left=None,right=None,wspace=None,hspace=None):
    # -- create subplots for blocks --
    gs_sub = GridSpec(nblocks,nblocks,hspace=hspace)
    gs_sub.update(left=left, right=right, wspace=wspace)
    ax = []
    for i in range(nblocks):
        ax_i = []
        for j in range(nblocks):
            ax_ij = plt.subplot(gs_sub[i,j])
            ax_i.append(ax_ij)
        ax.append(ax_i)
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

    # -- image to burst --
    burst = []
    for t in range(nframes):
        ox,oy = offset[t]
        x = nblocks//2 - ox
        y = nblocks//2 - oy
        image_t = image[ox:,oy:]
        burst.append(image_t)
    return burst

def dynamic():

    # -- settings --
    nblocks = 3
    patchsize = 5
    nframes = 3
    offset = np.array([[1,0],[0,0],[0,1]])
    ppf = 1
    burst = get_burst_data(nblocks,patchsize,nframes,offset,ppf)

    postfix = "optim"
    # bcolor = '#00FFFF' # blue/teal
    #bcolor = '#ffc107' # yellow
    bcolor = '#10b51c' # green
    create_plot_from_burst(burst,nblocks,nframes,patchsize,offset,postfix,bcolor)

    postfix = "rand"
    bcolor = 'red'
    offset = np.array([[0,1],[0,0],[1,0]])
    create_plot_from_burst(burst,nblocks,nframes,patchsize,offset,postfix,bcolor)


def create_plot_from_burst(burst,nblocks,nframes,patchsize,offset,postfix,bcolor):
    # -- create discrete colormap --
    # cmap = colors.ListedColormap(['blue', 'red','orange'])
    # cmap = colors.ListedColormap(['#000000', '#707070','#FFFFFF'])
    # cmap = colors.ListedColormap(['#000000', '#FFFFFF'])
    cmap = colors.ListedColormap(['#4630ff', '#8fc7ff'])
    bounds = list(np.arange(cmap.N+1))
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(8,3))
    fig.canvas.draw()

    # -- create subplots --
    ax = []
    pad = 0.05
    left,right = 0,1./nframes-pad
    for t in range(nframes):
        print(left,right)
        ax_t = create_gridspec(nblocks,left=left,right=right,wspace=0.05,hspace=0.00)
        ax.append(ax_t)
        left = right+pad
        right = 1./nframes + left - pad
    print(len(ax))
    print(len(ax[0]))
    print(len(ax[0][0]))

    # -- fill subplots for blocks --
    # gcolor = '#808080'
    gcolor = '#000000'
    for t in range(nframes):
        for i in range(nblocks):
            for j in range(nblocks):
                ax_i,ax_j = i,j
                data_t = burst[t]
                data_ij = get_data_ij(data_t,i,j,nblocks,patchsize)
                # print(ax[t][ax_i][ax_j])
                ax[t][ax_i][ax_j].imshow(data_ij, cmap=cmap, norm=norm)
                ax[t][ax_i][ax_j].grid(which='major', axis='both',
                                    linestyle='-', color=gcolor, linewidth=2)
                ax[t][ax_i][ax_j].set_xticks(np.arange(-.5, patchsize, 1));
                ax[t][ax_i][ax_j].set_yticks(np.arange(-.5, patchsize, 1));
                ax[t][ax_i][ax_j].set_xticklabels([])
                ax[t][ax_i][ax_j].set_yticklabels([])
                ax[t][ax_i][ax_j].set_aspect('equal')
                no_pointy_tics(ax[t][ax_i][ax_j])
                ox,oy = offset[t]
                x = nblocks//2 - ox
                y = nblocks//2 - oy
                if x == i and y == j:
                    boarder_axis(fig,ax[t][ax_i][ax_j],bcolor)

    fname = SAVE_DIR / f"dynamic_blocks_and_patches_{postfix}.png"
    plt.savefig(fname,transparent=True,bbox_inches='tight',dpi=300)
    plt.close("all")
    plt.clf()

def main():
    seed = 234
    np.random.seed(seed)
    torch.manual_seed(seed)
    dynamic()

if __name__ == "__main__":
    main()
