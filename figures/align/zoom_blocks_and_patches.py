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
    for tic in ax.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)

def get_burst_data(nblocks,patchsize,nframes,offset,ppf):

    # -- image --
    dyns = ppf*(nframes-1)
    zoom_size = 2*(nblocks//2) + patchsize + dyns
    c_size = 2*(nblocks//2) + patchsize
    image = np.random.rand(zoom_size, zoom_size) * 3

    # -- image to burst --
    burst = []
    for t in range(nframes):
        ox,oy = offset[t]
        x = nblocks//2 - ox
        y = nblocks//2 - oy
        image_t = image[ox:ox+c_size,oy:oy+c_size]
        burst.append(image_t)
    burst = np.stack(burst)
    return burst

def zoomed(burst,nblocks,patchsize):

    # zoom_size = 2*(nblocks//2) + patchsize
    # data = np.random.rand(zoom_size, zoom_size) * 3
    # print(data)
    nframes = burst.shape[0]
    data = burst[nframes//2]
    
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
    
    # -- create subplots for zoomed-in patch --
    gs_zoom = GridSpec(1,1)
    gs_zoom.update(left=0.0, right=0.18, wspace=0.05)
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
    h,w = data.shape
    ps = patchsize
    gcolor = '#000000'
    for i in range(nblocks):
        for j in range(nblocks):
            ax_i,ax_j = i,j
            data_ij = get_data_ij(data,i,j,nblocks,patchsize)
            # ax[ax_i][ax_j].imshow(data_ij, cmap=cmap, norm=norm)
            # ax[ax_i][ax_j].grid(which='major', axis='both',
            #                     linestyle='-', color=gcolor, linewidth=2)
            # ax[ax_i][ax_j].set_xticks(np.arange(-.5, patchsize, 1));
            # ax[ax_i][ax_j].set_yticks(np.arange(-.5, patchsize, 1));
            # ax[ax_i][ax_j].set_xticklabels([])
            # ax[ax_i][ax_j].set_yticklabels([])
            # ax[ax_i][ax_j].set_aspect('equal')
            # no_pointy_tics(ax[ax_i][ax_j])

    #---------------------------------------------------------
    #---------------------------------------------------------
    #---------------------------------------------------------


            # -- mask out all center frame except center patch --
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

            if j == nblocks//2 and i == (nblocks-1):
                ax[ax_i][ax_j].set_xlabel(f"Search Space of 5x5 Patches",fontsize=20)

            if j == nblocks//2 and i == 0:
                ax[ax_i][ax_j].set_xlabel(r"Frame $t$ Centered at Pixel $x$",fontsize=20)
                ax[ax_i][ax_j].xaxis.set_label_position('top') 


    #---------------------------------------------------------
    #---------------------------------------------------------
    #---------------------------------------------------------


    # -- plot middle of left-hand side --
    ax_i,ax_j = 1,0
    zoom_ax.axis('on')
    zoom_ax.imshow(data,cmap=cmap,norm=norm)
    zoom_ax.grid(which='major', axis='both', linestyle='-',
                 color=gcolor, linewidth=2)
    zoom_ax.set_xticks(np.arange(-.5, w, 1));
    zoom_ax.set_yticks(np.arange(-.5, h, 1));
    zoom_ax.set_xticklabels([])
    zoom_ax.set_yticklabels([])
    no_pointy_tics(zoom_ax)
    zoom_ax.set_xlabel("Zoomed\nView",fontsize=20)
    # tic.tick1On = tic.tick2On = False

    # -- draw arrows --
    # xy = (0.2, 0.2)
    # con = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data",
    #                       axesA=zoom_ax, axesB=ax[1][1])
    # zoom_ax.add_artist(con)

    plt.savefig(SAVE_DIR / "zoomed.png",bbox_inches='tight',transparent=True,dpi=300)

def main():

    # -- seed --
    seed = 234
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -- settings --
    nblocks = 3
    patchsize = 5
    nframes = 3
    offset = np.array([[1,0],[0,0],[0,1]])
    ppf = 1
    burst = get_burst_data(nblocks,patchsize,nframes,offset,ppf)

    zoomed(burst,nblocks,patchsize)

if __name__ == "__main__":
    main()
