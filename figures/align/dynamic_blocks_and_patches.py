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
    for tic in ax.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)

def boarder_axis(fig,ax,bcolor,x,y,ph,pw,ps,i,j):
    bbox = ax.get_tightbbox(fig.canvas.renderer)
    x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
    # slightly increase the very tight bounds:
    xpad = 0.05 * width
    ypad = 0.05 * height

    # -- colored around entire box --
    start_x = x0-0*xpad
    end_x = width+0*xpad
    start_y = y0-0*xpad
    end_y = height+0*ypad
    lw = 4

    # -- colored around selction only --
    # print(start_y,end_y,start_y < end_y,y0)
    # nblocks = 3
    # full_size = (ps + 2*(nblocks//2))
    # scale_ratio = ps/full_size

    # print("height,width", height, width)
    # print((3-1)-0.5-i,y0,width,pw,nblocks-i,i,y0)
    # ystart = y0
    # start_y_m = y0 + (nblocks-i - 0.5)*((height*scale_ratio)/full_size) #ystart / ph + 0.5/ph
    # end_y_m = height*scale_ratio#end_y#(ystart+ps) / ph + 0.5/ph
    # print(start_y,end_y," vs ",start_y_m,end_y_m)

    # #xstart / pw + 0.5/pw
    # print(i,j,x,y)
    # xstart = x0-j
    # start_x_m = x0 + (j+0*0.5)*((width*scale_ratio)/full_size)
    # print("start_x_m ",x0,start_x_m)
    # end_x_m = width*scale_ratio#end_y#(ystart+ps) / ph + 0.5/ph

    # lw = 2

    fig.add_artist(plt.Rectangle((start_x, start_y),
                                 end_x, end_y,
                                 edgecolor=bcolor, linewidth=lw, fill=False))

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

def dynamic():

    # -- settings --
    nblocks = 3
    patchsize = 5
    nframes = 3
    offset = np.array([[1,0],[0,0],[0,1]])
    ppf = 1
    burst = get_burst_data(nblocks,patchsize,nframes,offset,ppf)

    postfix = "optim"
    bcolor = '#00FFFF' # blue/teal
    #bcolor = '#ffc107' # yellow
    # bcolor = '#10b51c' # green
    create_plot_from_burst(burst,nblocks,nframes,patchsize,offset,postfix,bcolor)

    postfix = "rand"
    bcolor = 'red'
    offset = np.array([[0,1],[0,0],[1,0]])
    create_plot_from_burst(burst,nblocks,nframes,patchsize,offset,postfix,bcolor)


def create_plot_from_burst(burst,nblocks,nframes,patchsize,offset,postfix,bcolor):
    # -- create discrete colormap --
    # cmap = colors.ListedColormap(['blue', 'red','orange'])
    # cmap = colors.ListedColormap(['#000000', '#707070','#FFFFFF'])
    cmap = colors.ListedColormap(['#000000', '#FFFFFF'])
    # cmap = colors.ListedColormap(['#4630ff', '#8fc7ff'])
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
    h,w = burst[0].shape
    ps = patchsize
    # gcolor = '#808080'
    gcolor = '#000000'
    for t in range(nframes):
        for i in range(nblocks): # rows
            for j in range(nblocks): # columns
                # ax_i,ax_j = i,j

                # data_t = burst[t]
                # data_ij = get_data_ij(data_t,i,j,nblocks,patchsize)
                # # print(ax[t][ax_i][ax_j])
                # ax[t][ax_i][ax_j].imshow(data_ij, cmap=cmap, norm=norm)
                # ax[t][ax_i][ax_j].grid(which='major', axis='both',
                #                     linestyle='-', color=gcolor, linewidth=2)
                # ax[t][ax_i][ax_j].set_xticks(np.arange(-.5, patchsize, 1));
                # ax[t][ax_i][ax_j].set_yticks(np.arange(-.5, patchsize, 1));
                # ax[t][ax_i][ax_j].set_xticklabels([])
                # ax[t][ax_i][ax_j].set_yticklabels([])
                # ax[t][ax_i][ax_j].set_aspect('equal')
                # no_pointy_tics(ax[t][ax_i][ax_j])
                # ox,oy = offset[t]
                # x = nblocks//2 - ox
                # y = nblocks//2 - oy
                # if x == i and y == j:
                #     boarder_axis(fig,ax[t][ax_i][ax_j],bcolor)

                image_t = burst[t]
                ax_i,ax_j = i,j
                alphas = 0.15 * np.ones(image_t.shape)

                # -- mask out all center frame except center patch --
                include = not(t == nframes//2) or (i == nblocks//2 and j == nblocks//2)
                if include:
                    alphas[i:i+ps,j:j+ps] = 1.

                # data_ij = get_data_ij(data,i,j,patchsize)
                ax[t][ax_i][ax_j].imshow(image_t, alpha=alphas, cmap=cmap, norm=norm)
                # ax[ax_i][ax_j].grid(which='major', axis='both',
                #                     linestyle='-', color=gcolor, linewidth=2)
                xstart = -.5 + j
                ystart = -.5 + i
    
                # -- partial gridlines --
                xgrid = np.arange(xstart, xstart + patchsize+1, 1)
                ygrid = np.arange(ystart, ystart + patchsize+1, 1)
    
                if include:
                    for yk in ygrid:
                        xstart_m = xstart / w + 0.5/w
                        xend_m = (xstart+ps) / w + 0.5/w
                        ax[t][ax_i][ax_j].axhline(y=yk, xmin=xstart_m,
                                                  xmax=xend_m,
                                                  color=gcolor)
                    ystart = (nblocks-1)-0.5-i
                    for xk in xgrid:
                        ystart_m = ystart / h + 0.5/h
                        yend_m = (ystart+ps) / h + 0.5/h
                        ax[t][ax_i][ax_j].axvline(x=xk, ymin=ystart_m,
                                                  ymax=yend_m,
                                                  color=gcolor)

                # ax[ax_i][ax_j].set_xticks(np.arange(xstart, xstart + patchsize+1, 1));
                # ax[ax_i][ax_j].set_yticks(np.arange(ystart, ystart + patchsize+1, 1));
    
                ax[t][ax_i][ax_j].grid(which='minor', axis='both', alpha=0.15,
                                       linestyle='-', color=gcolor, linewidth=2)
                ax[t][ax_i][ax_j].set_xticks(np.arange(-.5, h, 1),minor=True);
                ax[t][ax_i][ax_j].set_yticks(np.arange(-.5, w, 1),minor=True);

                ax[t][ax_i][ax_j].set_xticklabels([])
                ax[t][ax_i][ax_j].set_yticklabels([])
                ax[t][ax_i][ax_j].set_aspect('equal')
                no_pointy_tics(ax[t][ax_i][ax_j])
    
                ox,oy = offset[t]
                x = nblocks//2 - ox
                y = nblocks//2 - oy
                if x == i and y == j:
                    if t == nframes//2: boarder_color = "#ffc107"
                    else: boarder_color = bcolor
                    boarder_axis(fig,ax[t][ax_i][ax_j],boarder_color,x,y,h,w,ps,i,j)

                if j == nblocks//2 and i == (nblocks-1):
                    ax[t][ax_i][ax_j].set_xlabel(f"Frame {t+1}",fontsize=20)
                    

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
