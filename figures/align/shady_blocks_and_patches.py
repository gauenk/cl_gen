"""

-- Initial Comments --

-> Center patch with arrows shooting out from center.

-> Having the arrows black are important.


White + Black + another bright color is a lot of jumps for the eyes
if colors are on the same end, lighter side, then they blend in

so maybe the pixels don't need to be so starkly contrasing.

be okay for the arrows to contrast from the pixels but 
the pixels themselves shouldn't be too loud.

pixels are noticably differet: 
example: white -> grey or black -> grey with opposite colored arrows

-- another idea --

animate the 7x7 patch to each 3x3 set of patches

-- another idea --





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

def get_data_ij(data,i,j,patchsize):
    ip = i# - nblocks//2
    jp = j# - nblocks//2
    si = ip
    ei = si+patchsize
    sj = jp
    ej = sj+patchsize
    return data[si:ei,sj:ej]

def no_pointy_tics(ax):
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


def shady():
    nblocks = 3
    patchsize = 5

    zoom_size = 2*(nblocks//2) + patchsize
    data = np.random.rand(zoom_size, zoom_size) * 3
    print(data)
    
    # -- create discrete colormap --
    # cmap = colors.ListedColormap(['red', 'blue', 'orange'])
    # cmap = colors.ListedColormap(['#3d3d3d', '#707070','#FFFFFF'])
    # cmap = colors.ListedColormap(['#FFFFFF', '#707070',])
    cmap = colors.ListedColormap(['#000000', '#FFFFFF'])

    # -- blues --
    # cmap = colors.ListedColormap(['#4630ff', '#5085fb','#8fc7ff'])
    # cmap = colors.ListedColormap(['#4630ff', '#8fc7ff'])
    # cmap = colors.ListedColormap(['#4630ff', '#ffffff'])

    bounds = list(np.arange(cmap.N+1))
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # -- create subplots for blocks --
    fig = plt.figure(figsize=(6,6))
    fig.canvas.draw()

    gs_sub = GridSpec(nblocks,nblocks)
    gs_sub.update(left=0.0, wspace=0.05, hspace=0.05)
    print(gs_sub.__getstate__())
    ax = []
    for i in range(nblocks):
        ax_i = []
        for j in range(nblocks):
            ax_ij = plt.subplot(gs_sub[i,j])
            ax_i.append(ax_ij)
        ax.append(ax_i)

    # -- fill subplots for blocks --
    # gcolor = '#808080'
    h,w = data.shape
    ps = patchsize
    gcolor = '#000000'
    for i in range(nblocks):
        for j in range(nblocks):
            ax_i,ax_j = i,j
            alphas = 0.15 * np.ones(data.shape)
            alphas[i:i+ps,j:j+ps] = 1.
            # data_ij = get_data_ij(data,i,j,patchsize)
            ax[ax_i][ax_j].imshow(data, alpha=alphas, cmap=cmap, norm=norm)
            # ax[ax_i][ax_j].grid(which='major', axis='both',
            #                     linestyle='-', color=gcolor, linewidth=2)
            xstart = -.5 + j
            ystart = -.5 + i

            # -- partial gridlines --
            xgrid = np.arange(xstart, xstart + patchsize+1, 1)
            ygrid = np.arange(ystart, ystart + patchsize+1, 1)

            for yk in ygrid:
                xstart_m = xstart / w + 0.5/w
                xend_m = (xstart+ps) / w + 0.5/w
                ax[ax_i][ax_j].axhline(y=yk, xmin=xstart_m,
                                       xmax=xend_m,
                                       color=gcolor)
            ystart = (nblocks-1)-0.5-i
            for xk in xgrid:
                ystart_m = ystart / h + 0.5/h
                yend_m = (ystart+ps) / h + 0.5/h
                ax[ax_i][ax_j].axvline(x=xk, ymin=ystart_m,
                                       ymax=yend_m,
                                       color=gcolor)

            # ax[ax_i][ax_j].set_xticks(np.arange(xstart, xstart + patchsize+1, 1));
            # ax[ax_i][ax_j].set_yticks(np.arange(ystart, ystart + patchsize+1, 1));

            ax[ax_i][ax_j].grid(which='minor', axis='both', alpha=0.15,
                                linestyle='-', color=gcolor, linewidth=2)
            ax[ax_i][ax_j].set_xticks(np.arange(-.5, h, 1),minor=True);
            ax[ax_i][ax_j].set_yticks(np.arange(-.5, w, 1),minor=True);

            ax[ax_i][ax_j].set_xticklabels([])
            ax[ax_i][ax_j].set_yticklabels([])
            ax[ax_i][ax_j].set_aspect('equal')
            no_pointy_tics(ax[ax_i][ax_j])


    plt.savefig(SAVE_DIR / "shady.png",bbox_inches='tight',transparent=True,dpi=300)

def main():
    shady()

if __name__ == "__main__":
    main()
