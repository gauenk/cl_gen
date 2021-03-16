# -- python imports --
import sys
sys.path.append("./lib")
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import pandas as pd

# -- project imports --
from pyutils.plot import add_legend
pd.set_option('display.max_row',1000)

def compute_sample_psnr(noise_level,n_frames,size):
    data = np.zeros(size)
    for n in range(n_frames):
        data += npr.normal(0,noise_level,size=size)
    data /= n_frames
    psnr = 10*np.log10(1./np.mean(data**2))
    return psnr

def main():
    print("HI")    
    
    frames = [1,2,3,5,8,10,15,20]
    # frames = [1,2,3,10,15,30,50]
    noise_levels = [1,3,5,10,15,25,50,75,100]
    size = 256*256
    repeats = 5
    ave_psnrs,std_psnrs = [],[]
    df = pd.DataFrame({"ave":[],"std":[],"noise_level":[],"frames":[]})
    for n_frames in frames:
        for noise_level in noise_levels:
            psnrs = []
            for r in range(repeats):
                psnrs.append(compute_sample_psnr(noise_level/255.,n_frames,size))
            ave_psnr = np.mean(psnrs)
            std_psnr = np.std(psnrs)
            df = df.append({"ave":ave_psnr,"std":std_psnr,
                            "noise_level":noise_level,"frames":n_frames},True)

    print(df)

    # -- plot data: frame label --
    fig,ax = plt.subplots(figsize=(8,8))
    frames = []
    for n_frames,df_nl in df.groupby("frames"):
        frames.append(str(n_frames))
        nl = df_nl['noise_level']
        ave = df_nl['ave']
        std = df_nl['std']
        ax.errorbar(nl,ave,yerr=std,label=f"{n_frames}")
    # plt.errorbar(noise_levels,ave_psnrs,yerr=std_psnrs,label="N=1")
    add_legend(ax,"Frames",frames,fontsize=12)
    ax.set_ylabel("PSNR",fontsize=12)
    ax.set_xlabel("Noise Level",fontsize=12)
    ax.set_title("Averaging Gaussian Samples")
    plt.savefig("./output/awgn_noise_levels.png",dpi=300)
    plt.clf()
    plt.cla()
    plt.close("all")


    # -- plot data: noise level label --
    fig,ax = plt.subplots(figsize=(8,8))
    nls = []
    for nl,df_frames in df.groupby("noise_level"):
        nls.append(str(nl))
        frames = df_frames['frames']
        ave = df_frames['ave']
        std = df_frames['std']
        ax.errorbar(frames,ave,yerr=std)
    add_legend(ax,"NoiseLevel",nls,fontsize=12)
    ax.set_ylabel("PSNR",fontsize=12)
    ax.set_xlabel("Frames",fontsize=12)
    ax.set_title("Averaging Gaussian Samples")
    plt.savefig("./output/awgn_frames.png",dpi=300)


if __name__ == "__main__":
    main()
