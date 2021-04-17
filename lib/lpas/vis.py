
# -- python imports --
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# -- project imports --
import settings
from pyutils.plot import add_legend

# -- setup path for vis --
ROOT_PATH_VIS = Path(f"{settings.ROOT_PATH}/output/lpas/vis/")
if not ROOT_PATH_VIS.exists(): ROOT_PATH_VIS.mkdir()
    
def explore_record(record):

    # order = ['score_function','patchsize','noise_type','nframes','nblocks','ppf']
    # patchsize = 13
    nframes = 3
    nblocks = 5
    ppf = 1.0
    score_function = ['ave','pairwise','refcmp']

    # record = record[record['patchsize'] == patchsize]
    record = record[record['nblocks'] == nblocks]
    record = record[np.isclose(record['ppf'].to_numpy(),ppf)]

    print("HI")
    # fig,ax = plt.subplot(figsize=(8,8))
    for patchsize,ps_df in record.groupby('patchsize'):
        for nframes,nframe_df in ps_df.groupby('nframes'):
            for score_fxn,score_df in nframe_df.groupby('score_function'):
                for noise_type,noise_df in score_df.groupby('noise_type'):

                    noise_index = noise_df['noisy_best_idx'].to_numpy().astype(np.int)
                    clean_index = noise_df['clean_best_idx'].to_numpy().astype(np.int)
                    align_index = noise_df['align_best_idx'].to_numpy().astype(np.int)
    
                    noise_psnr = np.mean(pdnumpy_to_ndnumpy(noise_df['noisy_psnr']),axis=0)
                    clean_psnr = np.mean(pdnumpy_to_ndnumpy(noise_df['clean_psnr']),axis=0)
                    align_psnr = np.mean(pdnumpy_to_ndnumpy(noise_df['align_psnr']),axis=0)
    
                    noise_acc = np.mean(noise_index == align_index)
                    clean_acc = np.mean(clean_index == align_index)
    
                    noise_score = noise_df['noisy_best_score']
                    clean_score = noise_df['clean_best_score']
                    align_score = noise_df['align_best_score']
    
                    noise_scores = load_numpy_along_series(noise_df['noisy_scores'])
                    clean_scores = load_numpy_along_series(noise_df['clean_scores'])
                    align_scores = load_numpy_along_series(noise_df['align_scores'])

                    # print("a",np.argsort(noise_scores[0])[0],noise_index)

                    sub = 2
                    C = 6*(75./255.)**2
                    noise_sort = np.argsort((noise_scores-C)**2,axis=1)[:sub,:100]
                    align_sort = np.argsort(align_scores,axis=1)[:sub,:100]
                    
                    acc,mindist = 0,0
                    for i in range(noise_sort.shape[0]):
                        mindist += np.min(np.abs(noise_sort[i] - align_index[i]))
                        acc += float(np.any(noise_sort[i] == align_index[i]))
                    acc /= noise_sort.shape[0]
                    mindist /= noise_sort.shape[0]
                    print("Noisy Type %s Catch It?: %2.3f | Min Dist: %2.3f" % (noise_type, acc, mindist) )
                    colors = np.arange(1,align_sort.shape[0]+1)[:,None] * np.ones_like(align_sort)
                    noise_sort = noise_sort.ravel().astype(np.float)
                    align_sort = align_sort.ravel().astype(np.float)
                    colors = colors.ravel().astype(np.int8)
                    fig,ax = plt.subplots(figsize=(8,8))
                    ax.plot([noise_index[:sub]],[noise_index[:sub]],'bx',markersize=15)
                    ax.plot([align_index[:sub]],[align_index[:sub]],'kx',markersize=15)
                    ax.scatter(noise_sort,align_sort,c=colors,cmap='prism')
                    stem = f"indice_plot_{int(patchsize)}_{int(nframes)}_{score_fxn}_{noise_type}.png"
                    plot_fn = Path(f"{ROOT_PATH_VIS}") / Path(stem)
                    plt.savefig(plot_fn)
                    plt.close(fig)
                    plt.close("all")
                    plt.clf()
                    plt.cla()
                    # print(align_sort.shape)
                    # print(np.max(noise_sort),np.max(align_sort))
                    # print("align",align_scores.shape)
                    # print("s",noise_sort.shape,noise_scores.shape)
                    # print(noise_sort,align_index)
                    # print("m",np.max(noise_sort),align_index[0])
                    # print(np.where(noise_sort[0] == align_index[0]))
                    # print(np.where(align_index == noise_sort))
                    # print(align_score)
                    # print(np.min(align_scores,axis=1))
                    # print(align_scores.shape)
                    # print(np.min(align_scores))
    
                    # print(nframes,score_fxn,noise_type,noise_acc,clean_acc)
                    # print(patchsize,nframes,score_fxn,noise_type,noise_acc,clean_acc)
                    # psnrs = np.array([noise_psnr,clean_psnr,align_psnr])
                    # print(psnrs)

                
    # record_3f = record[record['nframes'] == 3]
    # record_5f = record[record['nframes'] == 5]


def load_numpy_along_series(series):
    data = []
    for i in range(len(series)):
        data.append(np.load(series.iloc[i]))
    data = np.array(data)
    return data


def pdnumpy_to_ndnumpy(series):
    pdnumpy = series.to_numpy()
    ndnumpy = []
    for i in range(len(series)):
        ndnumpy.append(np.fromstring(pdnumpy[i][1:-1],sep=' '))
    ndnumpy = np.array(ndnumpy)
    return ndnumpy


def plot_ave_nframes_scoreacc_noisetype(record):
    
    nframes = 3
    nblocks = 5
    ppf = 1.0
    score_function = ['ave','pairwise','refcmp']
    noise_type_codes,noise_type_unique = pd.factorize(record['noise_type'])
    record['noise_type_codes'] = noise_type_codes

    # ['score_function','patchsize','noise_type','nframes','nblocks','ppf']

    record = record[record['nblocks'] == nblocks]
    record = record[np.isclose(record['ppf'].to_numpy(),ppf)]
    record = record[record['score_function'] == 'ave']
    record = record[record['patchsize'] == 13]
    

    fig,ax = plt.subplots(2,figsize=(8,8))
    labels = []
    for nframes,df in record.groupby('nframes'):

        # -- compute acc --

        tcodes,tnoises,noise_acc,clean_acc = [],[],[],[]
        for tnoise,noise_df in df.groupby("noise_type"):
            noise_index = noise_df['noisy_best_block']
            clean_index = noise_df['clean_best_block']
            align_index = noise_df['align_best_block']
            tcode = noise_df['noise_type_codes'].to_numpy()[0]
            noise_acc.append(np.mean(noise_index == align_index))
            clean_acc.append(np.mean(clean_index == align_index))
            tnoises.append(tnoise)
            tcodes.append(tcode)

        # -- plot --
        ax[0].plot(tcodes,noise_acc,label=f'{int(nframes)}')
        ax[1].plot(tcodes,clean_acc,label=f'{int(nframes)}')

        ax[0].set_xticks(tcodes)
        ax[0].set_xticklabels(tnoises)

        ax[1].set_xticks(tcodes)
        ax[1].set_xticklabels(tnoises)
        labels.append(str(int(nframes)))

    add_legend(ax[0],"Frames",labels)
    add_legend(ax[1],"Frames",labels)
    plot_fn = ROOT_PATH_VIS / "ave_nframes_scoreacc_noisetype.png"
    print(f"Saving figure to path [{plot_fn}]")
    plt.savefig(plot_fn)
    plt.close("all")

