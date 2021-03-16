# python imports
import sys
sys.path.append("./lib")
sys.path.append("./tools")
import numpy as np
import numpy.random as npr
import pandas as pd
from easydict import EasyDict as edict
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# project imports
import settings
from example_static import test_disent
from aws_denoising import get_denoising_cfg
from pyutils.misc import write_pickle,read_pickle
from pyutils.plot import add_legend


def plots_exp_v1():
    epochs = [0]
    skel = "/home/gauenk/Documents/experiments/cl_gen/output/n2nwl/dynamic_wmf/64_2_6/1/nonblind/3/25.0/sup_record_losses_{}.csv"
    fps = [skel.format(epoch) for epoch in epochs]
    pds = [pd.read_csv(fp) for fp in fps]

    fig,ax = plt.subplots()
    pds[0].reset_index().plot(x="index",y="ot",ax=ax)
    pds[0].reset_index().plot(x="index",y="kpn",ax=ax)
    pds[0].reset_index().plot(x="index",y="psnr",ax=ax)

    # pds[0].reset_index().plot(x="index",y="kpn",'-+')
    plt.savefig("./sup_ot_loss.png")

def plots_exp_v2():
    epochs = [0]
    skel = "/home/gauenk/Documents/experiments/cl_gen/output/n2n-kpn/dynamic_wmf/64_2_6/50000/nonblind/3/25.0/record_losses_26.csv"
    # skel = "/home/gauenk/Documents/experiments/cl_gen/output/n2n-kpn/dynamic_wmf/64_2_6/50000/nonblind/3/25.0/record_supOT_withweight_3.csv"
    # skel = "/home/gauenk/Documents/experiments/cl_gen/output/n2n-kpn/dynamic_wmf/64_2_6/50000/nonblind/3/25.0/record_withweight_15.csv"
    fps = [skel.format(epoch) for epoch in epochs]

    pds = [pd.read_csv(fp, index_col=0) for fp in fps]

    df = pds[0]
    df = df.reset_index(drop=True)
    # df = df.loc[:,~df.columns.str.contains('^psnr')]
    # df = df.loc[:,["ot_loss_rec_frame","ot_loss_rec_frame_w","ot_loss_raw_frame","ot_loss_raw_frame_w","kpn_loss","psnr_ave","psnr_std"]]
    df = df.loc[:,["ot_loss_rec_frame","ot_loss_raw_frame","kpn_loss"]]
    print(df)
    cols = list(df.columns)
    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_ylabel("loss")
    df.plot(ax=ax,logy=True)
    add_legend(ax,"Losses",cols)

    ax2 = ax.twinx()
    ax2.set_ylabel("psnr")
    psnr_ave = pds[0].loc[:,"psnr_ave"]
    ax2.plot(np.arange(len(psnr_ave)),psnr_ave,lw=2,color='black')
    cols += ['psnr']


    # pds[0].reset_index().plot(x="index",y="ot",ax=ax)
    # pds[0].reset_index().plot(x="index",y="kpn",ax=ax)
    # pds[0].reset_index().plot(x="index",y="psnr",ax=ax)

    # pds[0].reset_index().plot(x="index",y="kpn",'-+')
    plt.savefig("./sup_kpn_tracking.png",bbox_inches='tight',dpi=300)


def main():
    plots_exp_v2()

main()
