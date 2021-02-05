"""

spatio temporal kernel
the kernel says frame #5 i am frame #5 for 1 - 10
the 5th kernel is strong
others should be weatend to be k


yiheng
-> trained the res50; feed 10 frames in and dump 10 frames out
-> for testing compare all 10 frames

ask network to somehow regularize the weights so we on't allow the temporal axis to have 1 strong and remaining weak
if motion kernel aligned over the time, then we won't have blurring


the kernel is ee has a direction that is aligned with the motion then won't get any motion blur

-> output must be "clean image" for this to work


scheme 1:

-> ppf times direction
-> 100 frames with a total amount of shift.
-> don't worry about increasing noise as frame rate increases
-> horizontal motion 
-> fix number of total pixels shifted to 20 pixels over N frames.
-> then we should be able to see a pattern 
-> wthout anything we just get an indicator; we want to get a shifting behavior
-> trajectory for each kernel pixel



Regularize to ensure the filters have this trajectory
Feed the network more than just a single frame?
-> we have N frames per moving image with a middle reference
-> output we take the "closest" reference and compare with the set of noisy images. (1,2,4,5) all noisy and (3) noisy is output
-> is it possible to cycle through this process? I am not just interested in (3). I want it to also work for 1 or 2 or 4 or 5. We have the freedome
to use whatever frame we want to 
-> we get to know the ordering of the events
-> more images with fewer frames is better (unseen motion)
-> one model to predict the 2nd or 5th frame
-> 


-> another parameters to feed a 3 here then output is 3rd frame.
-> another parmaeter to feeda  3 and then output is 4th 
-> 10 frames in and 10 frames out. each frame is a denoised frame. 

3d convolution.

2 pixel shift from one to another

if 100 frames then shift 200 

100 frames network not scale so well. information overlap with 1st and 100th. but if 100 channel 

20 pixel shift over N frames

-> 0.5 pixels it can find the "edge" in the time dimension
-> 

whats happening is: 1st and 30th frame are shifted by 60 pixels
-> find something 60 pixels away from the 

"""


# python imports
import sys,csv
sys.path.append("./lib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# project imports
import settings
from pyutils.plot import add_legend

def read_file(path):
    data = None
    with open(path,"r") as f:
        reader = csv.reader(f,delimiter=',')
        data = next(iter(reader))
    return data


def load_noise_frame_grid():
    
    base = Path(f"{settings.ROOT_PATH}/output/n2n/")
    # noise_level_grid = [10,25,50,100,150,200,0]
    # noise_name_grid = ['g10','g25','g50','g100','g150','g200','msg']
    S = 50000
    tr_epochs = 30
    # noise_level_grid = [10,25,50,100,150,200]
    # noise_name_grid = ['g10','g25','g50','g100','g150','g200']
    noise_level_grid = [25]
    noise_name_grid = ['g25']
    blind_grid = ["blind","nonblind"]
    # blind_grid = ["blind"]
    # frame_grid = [2,3,4,5,10,20,30,100]
    # frame_grid = [4,50,100]
    frame_grid = [3,5,10]
    results = pd.DataFrame({'noise_level':[],'noise_name':[],'frames':[],'psnr':[],'epoch':[]})
    for noise,level in zip(noise_name_grid,noise_level_grid):
        for frame in frame_grid:
            for blind in blind_grid:
                # if blind == "blind": continue
                # stem = Path(f"{noise}/{S}/{blind}/{frame}.csv")
                stem = Path(f"./dynamic/128_2/{S}/{blind}/{frame}/{level}/results.csv")
                #stem = Path(f"{noise}/tr_epochs_{tr_epochs}/{blind}/{frame}.csv")
                print(stem)
                path = base / stem
                info = read_file(path)
                elem = {'noise_level':level,'noise_name':noise,'frames':frame,
                        'blind':blind,'psnr':float(info[2]),'epoch':int(info[1])}
                results = results.append(elem,ignore_index=True)
    return results

def plot_noise_grid(results):
    """
    plot sigma v.s. psnr with many "frames" lines
    """
    fig,ax = plt.subplots(figsize=(8,8))

    min_val,max_val = 1000,-1
    legend_str = []
    for frames,frames_df in results.groupby('frames'):
        frames_df = frames_df.sort_values('noise_level')
        noise_levels = frames_df['noise_level'].to_numpy()
        psnr = frames_df['psnr'].to_numpy()
        ax.plot(noise_levels,psnr)
        legend_str.append(str(int(frames)))
        print(psnr.min(),psnr.max())
        if min_val > psnr.min():
            min_val = psnr.min()
        if max_val < psnr.max():
            max_val = psnr.max()
    
    noise_levels = np.sort(results['noise_level'].unique())
    noise_names = ["{:d}".format(int(y)) for y in noise_levels]
    # noise_names[0] = 'msg'
    ax.set_xticks(noise_levels)
    ax.set_xticklabels(noise_names,fontsize=13)

    
    ytick_locs = np.linspace(min_val,max_val,4)
    ytick_names = ["{:2d}".format(int(y)) for y in ytick_locs]
    ax.set_yticks(ytick_locs)
    ax.set_yticklabels(ytick_names,fontsize=13)

    ax.set_xlabel("sigma",fontsize=13)
    ax.set_ylabel("PSNR",fontsize=13)
    ax.set_title("PSNR Across Noise Level",fontsize=15)
    add_legend(ax,"Frame Count",legend_str,shrink = True,fontsize=13,framealpha=1.0)
    plt.savefig(f"{settings.ROOT_PATH}/output/n2n/noise_grid.png")

def plot_groupby_noise_ax(results,ax,mrk='-x'):
    min_val,max_val = 1000,-1
    legend_str = []
    for noise,noise_df in results.groupby('noise_level'):
        noise_df = noise_df.sort_values('frames')
        frames = noise_df['frames'].to_numpy()
        psnr = noise_df['psnr'].to_numpy()
        ax.plot(np.log(frames),psnr,mrk)
        noise_name = noise_df['noise_name'].iloc[0]
        legend_str.append(noise_name[1:])
        print(psnr.min(),psnr.max())
        if min_val > psnr.min():
            min_val = psnr.min()
        if max_val < psnr.max():
            max_val = psnr.max()
    return legend_str,min_val,max_val

def plot_frame_grid(results):
    """
    plot frames v.s. psnr with many "sigma" lines
    """
    fig,ax = plt.subplots(figsize=(10,8))
    results = results.sort_values("noise_level")
    blind_results = results[results['blind'] == 'blind']
    nonblind_results = results[results['blind'] == 'nonblind']
    b_legend_str,b_min_val,b_max_val = plot_groupby_noise_ax(blind_results,ax,'-x')
    plt.gca().set_prop_cycle(None)
    nb_legend_str,nb_min_val,nb_max_val = plot_groupby_noise_ax(nonblind_results,ax,'--+')
    min_val = np.min([b_min_val,nb_min_val])
    max_val = np.max([b_max_val,nb_max_val])
    legend_str = [x+'-n' for x in b_legend_str] + [x + '-c' for x in nb_legend_str]

    
    frame_levels = np.sort(results['frames'].unique())
    frame_names = ["{:d}".format(int(y)) for y in frame_levels]
    ax.set_xticks(np.log(frame_levels))
    ax.set_xticklabels(frame_names,fontsize=13)
    
    ytick_locs = np.linspace(min_val,max_val,4)
    ytick_names = ["{:2d}".format(int(y)) for y in ytick_locs]
    ax.set_yticks(ytick_locs)
    ax.set_yticklabels(ytick_names,fontsize=13)

    ax.set_xlabel("log(N)",fontsize=13)
    ax.set_ylabel("PSNR",fontsize=13)
    ax.set_title("PSNR Across Frame Count",fontsize=15)
    add_legend(ax,"Noise Level",legend_str,shrink = True,fontsize=12,
               framealpha=1.0,ncol=2,shrink_perc=.80)
    plt.savefig(f"{settings.ROOT_PATH}/output/n2n/frame_grid.png")

def plot_noise_frame_grid():
    results = load_noise_frame_grid()
    blind_results = results[results['blind'] == 'blind']
    nonblind_results = results[results['blind'] == 'nonblind']
    plot_noise_grid(nonblind_results)
    # plot_frame_grid(nonblind_results)
    plot_frame_grid(results)

if __name__ == "__main__":
    print("HI")
    plot_noise_frame_grid()
