# -- python imports --
import math
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict

# -- plotting imports --
import matplotlib
import seaborn as sns
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

# -- project imports --
from pyplots.legend import add_legend
from pyplots.log import get_matplotlib_formatters
from datasets.wrap_image_data import load_resample_dataset,sample_to_cuda

def method_names(method):
    if method == "nnf":
        return "L2-Global"
    elif method == "nnf_local":
        return "L2-Local"
    elif method == "nvof":
        return "NVOF"
    elif method == "split":
        return "L2-Global (Noisy)"
    elif method == "ave_simp":
        return "L2-Local (Noisy)"
    elif method == "est":
        return "Ours (Noisy)"
    else:
        return method.replace("_","-")

def create_quality_v_runtime_plot(records,egrids,exp_cfgs):

    # -- get relevant data --
    data = extract_formatted_data(records,egrids,exp_cfgs)

    # -- create scatter plot --
    fig,ax = plt.subplots(figsize=(8,8))
    ax.tick_params(axis='both', which='major', labelsize=20, pad=8)
    ax.tick_params(axis='both', which='minor', labelsize=20, pad=8)
    ax.set_xscale('log')
    plt_scat = ax.scatter(10**data['ave_log_runtime'],
                          data['ave_psnr'],
                          marker='o',
                          s=120)
    ax.set_xlabel("Performance (miliseconds)",fontsize=20,labelpad=10)
    ax.set_ylabel("Denoiser Quality (PSNR)",fontsize=20,labelpad=10)

    # -- set xticks --
    # maj_fmt,maj_loc,min_loc = get_matplotlib_formatters()
    # ax.xaxis.set_major_formatter(maj_fmt)
    # ax.xaxis.set_major_locator(maj_loc)
    # ax.xaxis.set_minor_locator(min_loc)

    # -- set xlims --
    # xmin = data['ave_runtime'].min()
    # xlim_min = xmin - .1
    # xlim_max = ax.get_xlim()[1] + 100.
    # ax.set_xlim((xlim_min,xlim_max))

    # -- get scatter plot sample coordiantes --
    offset = .15
    text_len = 1.10
    axis_lims = (ax.get_xlim(),ax.get_ylim())
    xvalues = data['ave_log_runtime'].to_numpy()
    print(xvalues.min(),xvalues.max())
    log_xvalues = data['ave_log_runtime'].to_numpy()
    yvalues = data['ave_psnr'].to_numpy()
    for idx, method in enumerate(data['method']):
        # print(method)
        # print(data['ave_psnr'][idx],data['ave_runtime'][idx],data['method'][idx])
        # x = data['ave_log_runtime'][idx]+offset
        # x = data['ave_runtime'][idx]+offset
        # y = data['ave_psnr'][idx]+offset
        if len(method) < 7:
            text_len = 0.4
        elif len(method) < 10:
            text_len = 0.8
        else:
            text_len = 1.1
        logx,x,y = log_xvalues[idx],xvalues[idx],yvalues[idx]
        xoffset,yoffset = get_text_offset(logx,10**x,y,offset,text_len,axis_lims)
        text_x = xoffset
        ax.text(text_x,y+yoffset,method,fontsize=25)

    # ax.set_xscale('log')

    ax.set_title("Quality and Performance Trade-off",fontsize=30)
    fn = "./psnr_vs_runtime.png"
    plt.savefig(fn,dpi=300,bbox_inches='tight')
    print(f"Wrote [quality_v_runtime] plot at {fn}")

    plt.clf()
    plt.cla()
    plt.close("all")

def get_text_offset(logx,x,y,offset,text_len,axis_lims):

    # -- init and unpack --
    xlims = np.array(axis_lims[0])
    ylims = np.array(axis_lims[1])

    # -- select quandrant --
    left_right = np.argmin(np.abs(logx - np.log10(xlims)))
    bottom_top = np.argmin(np.abs(y - ylims))

    # -- modify xtext --
    xoffset = 0
    if left_right == 0: # closest to left
        xoffset = offset
    else: # right
        xoffset = -(offset + text_len)

    # -- from standard to log offset --
    xoffset = 10**(logx + xoffset)

    # -- modify ytext --
    yoffset = 0
    if bottom_top == 0: # closest to bottom
        yoffset = 2*offset
    else: # top
        yoffset = -5*offset

    return xoffset,yoffset

def extract_formatted_data(records,egrids,exp_cfgs):
    # -- plot accuracy of methods  --
    fmt_data = []
    skip_methods = ["of","ave"]
    for method,mgroup in records.groupby('methods'):
        if method in skip_methods: continue
        smethod = method_names(method)
        # smethod = method.replace("_","-")
        psnrs = []

        # -- gather psnrs into array --
        for mpsnrs in mgroup['psnrs']:
            psnrs.extend(list(mpsnrs))
        psnrs = np.ma.masked_invalid(psnrs).filled(0.)
        ave_psnr = np.mean(psnrs)
        runtimes = mgroup['runtimes'].to_numpy()*1000.
        ave_runtime = np.mean(runtimes)+1e-8
        # Convert to Miliseconds, so +3 
        log_runtimes = (np.ma.log(runtimes)/np.log(10.)+3.).filled(0.)
        ave_log_runtime = np.mean(log_runtimes)
        print(method,ave_psnr,ave_runtime)
        fmt_data.append({'method':smethod,
                         'ave_psnr':ave_psnr,
                         'ave_runtime':ave_runtime,
                         'ave_log_runtime':ave_log_runtime})
    fmt_data = pd.DataFrame(fmt_data)
    return fmt_data

def extract_formatted_data_groupby_nframes_std(records,egrids,exp_cfgs):
    # -- plot accuracy of methods  --
    fmt_data = []
    for nframes,fgroup in records.groupby('nframes'):
        handles,labels = [],[]
        for std,std_group in fgroup.groupby('std'):
            fig,ax = plt.subplots(figsize=(8,4))
            sstd = str(std).split(".")[0]
            for method,mgroup in std_group.groupby('methods'):
                smethod = method.replace("_","-")
                psnrs = []
                for mpsnrs in mgroup['psnrs']:
                    psnrs.extend(list(mpsnrs))
                ave_psnr = np.mean(psnrs)
                runtimes = mgroup['runtimes'].to_numpy()
                ave_runtime = np.mean(runtimes)
                fmt_data.append({'method':method,
                                 'ave_psnr':ave_psnr,
                                 'ave_runtime':ave_runtime,
                                 'nframes':nframes,
                                 'std':std})
    fmt_data = pd.DataFrame(fmt_data)
    return fmt_data

def remove_middle(m_scores,method):
    npix,nframes = m_scores.shape
    if method == "ave":
        ref_t = (nframes-1)//2
        m_scores = np.concatenate([m_scores[:,:ref_t],m_scores[:,ref_t+1:]],axis=1)
    elif method == "est":
        ref_t = nframes//2
        m_scores = np.concatenate([m_scores[:,:ref_t],m_scores[:,ref_t+1:]],axis=1)
    else:
        raise ValueError(f"Uknown method {method}")
    return m_scores

def create_ideal_v_noise_plot(records,egrids,exp_cfgs):
    # create_ideal_v_noise_plot_v1(records,egrids,exp_cfgs)
    create_ideal_v_noise_plot_v2(records,egrids,exp_cfgs)

def create_ideal_v_noise_plot_v1(records,egrids,exp_cfgs):

    # -- plot accuracy of methods  --
    for nframes,fgroup in records.groupby('nframes'):
        handles,labels = [],[]
        for std,std_group in fgroup.groupby('std'):
            fig,ax = plt.subplots(figsize=(8,4))
            sstd = str(std).split(".")[0]
            for method,mgroup in std_group.groupby('methods'):

                if not(method in ["ave","est"]): continue
                smethod = method.replace("_","-")

                scores = []
                for m_scores in mgroup['optimal_scores']:
                    m_scores = remove_middle(m_scores,method)
                    scores.extend(list(m_scores))
                # ave_psnr = np.mean(psnrs)
                print("method",method,"std",sstd,"m_scores",len(scores))
                scores = np.array(scores)
                label = smethod
                # handle = ax.hist(psnrs,label=label)
                
                handle = sns.histplot(scores,label=label,ax=ax)
                labels.append(label)
                handles.append(handle)

            title = f"Historgram of Ideal Scores [nframes: {nframes}] [std: {sstd}]"
            ax.set_title(title)
            # add_legend(ax,title,labels,None,shrink = True,
            #            fontsize=15,framealpha=1.0,ncol=1,shrink_perc=.80)
            save_dir = Path("./")
            fn =  save_dir / f"./hist_scores_v_noise_v1_{nframes}_{sstd}.png"
            print(f"Wrote plot to [{fn}]")
            plt.savefig(fn,transparent=False,bbox_inches='tight',dpi=300)
            plt.close('all')

                
def create_ideal_v_noise_plot_v2(records,egrids,exp_cfgs):


    # -- plot accuracy of methods  --
    for method,mgroup in records.groupby('methods'):
        handles,labels = [],[]
        if not(method in ["ave","est"]): continue
        # if not(method in ["ave"]): continue
        smethod = method.replace("_","-")

        for nframes,fgroup in mgroup.groupby('nframes'):

            fig,ax = plt.subplots(figsize=(8,4))

            for std,std_group in fgroup.groupby('std'):

                sstd = str(std).split(".")[0]
                scores = []
                for f_scores in std_group['optimal_scores']:
                    print(np.amin(f_scores,0))
                    f_scores = remove_middle(f_scores,method)
                    print(np.amin(f_scores,0))
                    scores.extend(list(f_scores))
                # ave_psnr = np.mean(psnrs)

                print("method",method,"std",sstd,"m_scores",len(scores))
                scores = np.array(scores)
                label = smethod
                # handle = ax.hist(psnrs,label=label)
                
                handle = sns.histplot(scores,label=label,ax=ax)
                labels.append(label)
                handles.append(handle)

            title = f"Historgram of Ideal Scores [method: {smethod}] [nframes: {nframes}]"
            ax.set_title(title)
            # add_legend(ax,title,labels,None,shrink = True,
            #            fontsize=15,framealpha=1.0,ncol=1,shrink_perc=.80)
            save_dir = Path("./")
            fn =  save_dir / f"./hist_scores_v_noise_v2_{nframes}_{smethod}.png"
            print(f"Wrote plot to [{fn}]")
            plt.savefig(fn,transparent=False,bbox_inches='tight',dpi=300)
            plt.close('all')
