# -- python imports --
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict

# -- plotting imports --
import matplotlib
import seaborn as sns
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

# -- project imports --
from pyplots.legend import add_legend
from datasets.wrap_image_data import load_resample_dataset,sample_to_cuda

def create_quality_v_noise_plot(records,egrids,exp_cfgs):

    # -- plot accuracy of methods  --
    for nframes,fgroup in records.groupby('nframes'):
        handles,labels = [],[]
        for std,std_group in fgroup.groupby('std'):
            fig,ax = plt.subplots(figsize=(8,4))
            sstd = str(std).split(".")[0]
            for method,mgroup in std_group.groupby('methods'):
                if method != "ave": continue
                smethod = method.replace("_","-")
                psnrs = []
                for mpsnrs in mgroup['nnf_acc']:
                    psnrs.extend(list(mpsnrs))
                # ave_psnr = np.mean(psnrs)
                print("method",method,"std",sstd,"psnrs",len(psnrs))
                psnrs = np.array(psnrs)
                label = smethod
                # handle = ax.hist(psnrs,label=label)
                
                handle = sns.histplot(psnrs,label=label,ax=ax)
                labels.append(label)
                handles.append(handle)

            title = "Accuracy Historgrams [nframes: {nframes}] [std: {sstd}]"
            ax.set_title(title)
            # add_legend(ax,title,labels,None,shrink = True,
            #            fontsize=15,framealpha=1.0,ncol=1,shrink_perc=.80)
            save_dir = Path("./")
            fn =  save_dir / f"./hist_acc_v_noise_{nframes}_{sstd}.png"
            print(f"Wrote plot to [{fn}]")
            plt.savefig(fn,transparent=False,bbox_inches='tight',dpi=300)
            plt.close('all')
    
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
