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

def get_data_ij(data,i,j,nblocks,patchsize):
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

def zoomed():
    nblocks = 3
    patchsize = 5

    zoom_size = 2*(nblocks//2) + patchsize
    data = np.random.rand(zoom_size, zoom_size) * 3
    print(data)
    
    # -- create discrete colormap --
    # cmap = colors.ListedColormap(['red', 'blue', 'orange'])
    # cmap = colors.ListedColormap(['#3d3d3d', '#707070','#FFFFFF'])
    cmap = colors.ListedColormap(['#FFFFFF', '#707070',])
    # cmap = colors.ListedColormap(['#000000', '#FFFFFF'])

    # -- blues --
    # cmap = colors.ListedColormap(['#4630ff', '#5085fb','#8fc7ff'])
    # cmap = colors.ListedColormap(['#4630ff', '#8fc7ff'])
    # cmap = colors.ListedColormap(['#4630ff', '#ffffff'])

    bounds = list(np.arange(cmap.N+1))
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # -- create subplots for zoomed-in patch --
    gs_zoom = GridSpec(1,1)
    gs_zoom.update(left=0.0, right=0.25, wspace=0.05)
    zoom_ax = plt.subplot(gs_zoom[0,0])

    # -- create subplots for blocks --
    gs_sub = GridSpec(nblocks,nblocks)
    gs_sub.update(left=0.30, wspace=0.05, hspace=0.05)
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
    gcolor = '#000000'
    for i in range(nblocks):
        for j in range(nblocks):
            ax_i,ax_j = i,j
            data_ij = get_data_ij(data,i,j,nblocks,patchsize)
            ax[ax_i][ax_j].imshow(data_ij, cmap=cmap, norm=norm)
            ax[ax_i][ax_j].grid(which='major', axis='both',
                                linestyle='-', color=gcolor, linewidth=2)
            ax[ax_i][ax_j].set_xticks(np.arange(-.5, patchsize, 1));
            ax[ax_i][ax_j].set_yticks(np.arange(-.5, patchsize, 1));
            ax[ax_i][ax_j].set_xticklabels([])
            ax[ax_i][ax_j].set_yticklabels([])
            ax[ax_i][ax_j].set_aspect('equal')
            no_pointy_tics(ax[ax_i][ax_j])

    # -- plot middle of left-hand side --
    ax_i,ax_j = 1,0
    zoom_ax.axis('on')
    zoom_ax.imshow(data,cmap=cmap,norm=norm)
    zoom_ax.grid(which='major', axis='both', linestyle='-',
                 color=gcolor, linewidth=2)
    zoom_ax.set_xticks(np.arange(-.5, zoom_size, 1));
    zoom_ax.set_yticks(np.arange(-.5, zoom_size, 1));
    zoom_ax.set_xticklabels([])
    zoom_ax.set_yticklabels([])
    no_pointy_tics(zoom_ax)
    # tic.tick1On = tic.tick2On = False

    # -- draw arrows --
    # xy = (0.2, 0.2)
    # con = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data",
    #                       axesA=zoom_ax, axesB=ax[1][1])
    # zoom_ax.add_artist(con)

    plt.savefig(SAVE_DIR / "zoomed.png",bbox_inches='tight',transparent=True,dpi=300)

def main():
    zoomed()

if __name__ == "__main__":
    main()
