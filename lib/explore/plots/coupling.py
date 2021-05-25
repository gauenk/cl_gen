"""
Create plots associated with the coupling phenomena among frames
"""

# -- python imports --
import numpy as np
import numpy.random as npr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange

# -- pytorch imports --
import torch
import torch.nn.functional as f

# -- project imports --


def jitter(ndarray):
    ratio = ndarray.max() - ndarray.min()
    return npr.normal(ndarray,scale=0.1 * ratio)

def normalize_row(tensor):
    return f.normalize(tensor,p=2,dim=0)

def update_ax_xgrid(ax_xgrid,xgrid_i):
    if len(ax_xgrid) == 0: return list(xgrid_i)
    xgrid_i = list(xgrid_i)
    ax_xgrid.extend(xgrid_i)
    ax_xgrid = np.unique(ax_xgrid)
    np.sort(ax_xgrid)
    ax_xgrid = list(ax_xgrid)
    return ax_xgrid

def plot_per_image(records,xgrids,sgrids,scores,scores_t,
                   bss,field,batch_size,normalize=True):
    B = batch_size
    scores = rearrange(scores,'bss (rb b) t -> t rb b bss',b=batch_size)
    rB = bss.shape[0]
    ax_xgrid = []
    for rb in range(rB):
        fig,ax = plt.subplots()
        for xgrid,sgrid in zip(xgrids[rb],sgrids[rb]):
            scores_rb = scores[:,rb,:,sgrid]
            for j in range(scores_rb.shape[1]):
                scores_i = torch.mean(scores_rb[:,j],dim=(0,)).numpy()
                if normalize:
                    scores_i -= np.min(scores_i)
                    scores_i /= np.max(scores_i)
                order = np.argsort(xgrid)
                xgrid_i = xgrid[order]
                scores_i = scores_i[order]
                ax_xgrid = update_ax_xgrid(ax_xgrid,xgrid_i)
                ax.plot(xgrid_i,scores_i,'x-')
        ax.set_xticks(ax_xgrid)
        ax.set_xticklabels([str(x) for x in ax_xgrid])
        # -- write to file --
        base = Path("./ave_frame_index_v_rest") / field / f"./{rb}b"
        if not base.exists(): base.mkdir(parents=True)
        fn = base / create_config_string(records)
        plt.savefig(fn,dpi=300,transparent=False,bbox_inches='tight')

        plt.clf()
        plt.cla()
        plt.close("all")
        
def init_new_xgrids(aligned,ax_xgrid):
    for x in ax_xgrid:
        if not (str(x) in aligned.keys()):
            aligned[str(x)] = []

def append_xgrid_values(aligned_dict,ax_xgrid,scores_sort):
    for x,s in zip(ax_xgrid,scores_sort):
        aligned_dict[str(x)].append(s)

def compute_aligned_dict_stats(aligned_dict):
    ave,std = [],[]
    for x,s in aligned_dict.items():
        s = np.stack(s)
        # print(len(s))
        # for i,s_i in enumerate(s):
        #     print(i,len(s_i))
        ave.append(np.mean(s))
        std.append(np.std(s))
    return ave,std

def plot_ave_image(records,xgrids,sgrids,scores,scores_t,
                   bss,field,batch_size,normalize,vectorized):

    # -- temp. fix for error in experiment --
    rep = scores.shape[1] // bss.shape[0]
    xgrids = np.repeat(xgrids,rep,axis=0)
    sgrids = np.repeat(sgrids,rep,axis=0)

    # -- align the scores --
    ax_xgrid = []
    aligned_dict = {}
    aligned_xgrids = []
    aligned = []
    scores = rearrange(scores,'bss img p -> img bss p')
    for i,(xgrid_i,sgrid_i) in tqdm(enumerate(zip(xgrids,sgrids))):
        for xgrid_b,sgrid_b in zip(xgrid_i,sgrid_i):

            # -- sort according to xgrid --
            order = np.argsort(xgrid_b)
            xgrid_sort = xgrid_b[order]
            scores_sort = scores[i,sgrid_b][order]

            # -- update unique xgrid values --
            ax_xgrid = update_ax_xgrid(ax_xgrid,xgrid_sort)
            init_new_xgrids(aligned_dict,ax_xgrid)
            append_xgrid_values(aligned_dict,ax_xgrid,scores_sort)

            # -- vectorized option --
            aligned_xgrids.append(xgrid_sort)
            aligned.append(scores_sort)

    if vectorized:
        aligned = torch.mean(torch.stack(aligned),dim=2)
        aligned_xgrids = np.stack(aligned_xgrids,axis=0)

    # -- list of plots --
    PLOT_MAX = 1000
    L = min([len(aligned),1000])
    if normalize: L = len(aligned)
    fig,ax = plt.subplots()
    for i in tqdm(range(L)):
        if normalize:
            aligned[i] -= torch.min(aligned[i])
            aligned[i] /= torch.max(aligned[i])
        if i >= PLOT_MAX: continue
        plot_aligned = jitter(aligned[i])
        ax.plot(aligned_xgrids[i],plot_aligned,alpha=0.05)

    # -- compute stats using dict --
    if vectorized:
        ave = torch.mean(aligned,dim=0)
        std = torch.std(aligned,dim=0)
    else:
        ave,std = compute_aligned_dict_stats(aligned_dict)
    ax.errorbar(ax_xgrid,ave,yerr=std)
    ax.set_xticks(ax_xgrid)
    ax.set_xticklabels([str(x) for x in ax_xgrid])
    
    # -- write to file --
    base = Path("./ave_frame_index_v_rest") / field
    if not base.exists(): base.mkdir(parents=True)
    fn = base / create_config_string(records)
    plt.savefig(fn,dpi=300,transparent=True,bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close("all")

def create_config_string(config):
    nframes = config['nframes']
    nblocks = config['nblocks']
    npatches = config['npatches']
    patchsize = config['patchsize']
    noise_type = config['noise_type']
    random_seed = config['random_seed']
    fn = f"{nframes}f_{nblocks}b_{npatches}p_{patchsize}ps_{random_seed}rs_{noise_type}.png"
    return fn

def reshape_scores(bss,bss_ibatch,scores,scores_t):
    xgrids,sgrids = bss_groups(bss,bss_ibatch,0)
    # -- "accurate": percent time true optima was reported optima? --
    # -- "coupling": influence of other frame aragements on a fixed frame search  --
       # a) along a grid, compute percent of time the correct
       # individual frame optima was equal to optima for various arangements of other frames.
       

def compute_coupling(configs,score_field):
    scores,scores_t,bss = [],[],[]
    for ridx,elem in configs.iterrows():
        elem = elem.to_dict()
        e_scores = elem[score_field]['scores']
        e_scores_t = elem[score_field]['scores_t']
        e_bss = elem['bss']
        e_bss_ibatch = elem['bss_ibatch']
        e_scores,e_scores_t = reshape_scores(bss,bss_ibatch,scores,scores_t)
        scores.append(e_scores)
        scores_t.append(e_scores_t)
    scores = torch.cat(scores,dim=0)
    scores_t = torch.cat(scores_t,dim=0)

    print(scores.shape)
    print(scores_t.shape)
    # scores = rearrange(records[exp_int][field]['scores'],'p ib ss -> ss ib p')
    # scores_t = rearrange(records[exp_int][field]['scores_t'],'p ib ss t -> ss t ib p')

def plot_score_v_frames(cfg,records,exp_fields):

    # -- plot couplings --
    score_field = "pixel_ave"
    coup_df = []
    records = pd.DataFrame(records)
    for nframes,nframes_df in records.groupby('nframes'):
        for noise,noise_df in nframes_df.groupby('noise_type'):
            coupling = compute_coupling(noise_df,score_field)
            elem = {'nframes':nframes,'noise':noise,'coupling':coupling}
            coup_df.append(elem)
    coup_df = pd.DataFrames(coup_df)
    print(coup_df)

    # -- plot couplings --
    # for noise,noise_df in coup_df.groupby('noise_type'):
    #     nframes = coup_df['nframes']
    #     couplings = coup_df['couplings']


def plot_frame_index_v_remaining_fixed(cfg,records,exp_fields,exp_int):

    # -- extract info --
    for exp_field in exp_fields:
        if exp_field == 'bss': continue
        print(exp_field,records[exp_int][exp_field])
    record = records[exp_int]
    bss = records[exp_int]['bss']
    bss_ibatch = records[exp_int]['bss_ibatch']
    field = 'pixel_ave'
    scores = rearrange(records[exp_int][field]['scores'],'p ib ss -> ss ib p')
    scores_t = rearrange(records[exp_int][field]['scores_t'],'p ib ss t -> ss t ib p')
    batch_size = records[exp_int]['batch_size']
    SS,IB,P = scores.shape
    # npatches, image batch, search_space batch

    # -- plot params --
    vectorized = False
    normalize = False

    # -- create plot grids --
    xindex = 0
    xgrids,sgrids = bss_groups(bss,bss_ibatch,xindex)
    igrid = np.arange(bss.shape[0])

    # -- plot per image --
    plot_per_image(record,xgrids,sgrids,scores,scores_t,
                   bss,field,batch_size,normalize)

    # -- ave score over images --
    plot_ave_image(record,xgrids,sgrids,scores,scores_t,bss,field,
                   batch_size,normalize,vectorized)
        
def bss_groups(bss,bss_ibatch,xindex):
    iB = bss.shape[0]
    xgrids,sgrids = [],[]
    for b in range(iB):
        xgrids_b,sgrids_b = bss_batch_groups(bss[b],xindex)
        xgrids.append(xgrids_b)
        sgrids.append(sgrids_b)
    # if vectorized:
    #     xgrids = np.stack(xgrids)
    #     sgrids = np.stack(sgrids)
    return xgrids,sgrids

def bss_batch_groups(bss,xindex):
    # -- convert int to strings --
    fmt = []
    for i,row in enumerate(bss):
        xvalue = row[xindex]
        rest = np.r_[row[:xindex],row[xindex+1:]]
        rest_str = "_".join([str(x) for x in rest])
        fmt.append({'xvalue':xvalue,'rest':rest_str})
    fmt = pd.DataFrame(fmt)
        
    # -- group by rest --
    xgrids,sgrids = [],[]
    for rest,group in fmt.groupby("rest"):
        xgrid = np.array([int(x) for x in group['xvalue']])
        sgrid = np.array(group.index)
        xgrids.append(xgrid)
        sgrids.append(sgrid)
    # if vectorized:
    #     xgrids = np.stack(xgrids)
    #     sgrids = np.stack(sgrids)
    return xgrids,sgrids
        
def col_indices(tensor,value):
    deltas = np.sum(torch.abs(tensor - value),axis=1)
    args = np.nonzero(deltas == 0)
    return args
