"""
Assess the quality of the 
global optima using a specific score
function 
by comparing with
the true global optima (no dynamics)

"""

# -- python imports --
import numpy as np
import numpy.random as npr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange
from easydict import EasyDict as edict
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# -- pytorch imports --
import torch
import torch.nn.functional as f
import torchvision.transforms.functional as tvF

# -- explore imports --
from settings import ROOT_PATH
from pyutils import add_legend
from explore.plots.utils import move_column_to_last,find_optima_index,get_pixel_fields,get_flownet_fields


def compare_global_optima_quality(cfg,records,exp_fields):
    quality = []
    for exp_int in range(len(records)):
        quality_i = experiment_global_optima_quality(cfg,records,
                                                     exp_fields,exp_int)
        
        # -- append extra info
        quality_df = pd.DataFrame(quality_i).T
        record = records[exp_int]
        fields = ['nframes',
                  'nblocks',
                  'noise_type',
                  'patchsize',
                  'random_seed']
        for field in fields:
            quality_df[field] = record[field]

        # -- format pandas results --
        quality_df.reset_index(inplace=True)
        quality_df = quality_df.rename(columns={'index':'metric'})
        quality.append(quality_df)

    quality = pd.concat(quality)
    quality = move_column_to_last(quality,"random_seed")
    quality = move_column_to_last(quality,"correct_v")

    print(quality)
    fields = get_pixel_fields(records[0])
    fields += get_flownet_fields(records[0])
    for field in fields:
        quality_metric = quality[quality['metric'] == field]
        print(field,quality_metric['acc'].mean())

    # -- stratify by noise level --
    optima_quality_by_noise_level(quality,fields)
            
    return quality

def argsort_noise_str(noises):
    levels = []
    for noise in noises:
        level = float(noise.split("-")[1].replace("p","."))
        levels.append(level)
    return np.argsort(levels)

def optima_quality_by_noise_level(quality,fields,postfix="default"):
    for noise,noise_df in quality.groupby("noise_type"):
        print(f"Noise: [{noise}]")
        for field in fields:
            quality_metric = noise_df[noise_df['metric'] == field]
            print(field,quality_metric['acc'].mean())

    optima_quality_by_noise_level_df(quality,fields,postfix="default")
    for nframes,df in quality.groupby("nframes"):
        optima_quality_by_noise_level_df(df,fields,postfix=nframes)

def optima_quality_by_noise_level_df(quality,fields,postfix="default"):

    # -- init plot --
    fig,ax = plt.subplots(figsize=(8,4))
    labels = []

    for field in fields:
        qmetric = quality[quality['metric'] == field]
        #quality_metric = noise_df[noise_df['metric'] == field]
        aves,stds,noises = [],[],[]
        for noise,noise_df in qmetric.groupby("noise_type"):
            noises.append(noise)
            aves.append(noise_df['acc'].mean())
            std = noise_df['acc'].std()
            N = len(noise_df['acc'])
            stderr = std / N
            stds.append(stderr)

        # -- reorder x-axis --
        order = argsort_noise_str(noises)
        noises = np.array(noises)[order]
        aves = np.array(aves)[order]
        stds = np.array(stds)[order]
        
        # -- plot --
        ax.errorbar(noises,aves,yerr=stds,marker='x',ls='-')
        labels.append(field)

    # -- include legend --
    def convert_nnf_labels(labels):
        convert = {'pixel_ave':'split',
                   'pixel_boostrapping':'ours',
                   'pixel_ransac':'ours_old'}
        new = []
        for label in labels:
            new_label = convert[label]
            new.append(new_label)
        return new
    labels = convert_nnf_labels(labels)
    ax = add_legend(ax,'Functions',labels,shrink_perc=.9,framealpha=0.0)
    ax.set_title("Experimental NNF Model Quality")
    ax.set_xlabel("Gaussian Noise Level")
    ax.set_ylabel("Accuracy of Model's Optima Compared to Actual NNF")

    # -- write to file --
    base = Path(ROOT_PATH) / "./output/explore/lpas/"
    if not base.exists(): base.mkdir(parents=True)
    fn = base / f"optima_quality_v_noise_level_{postfix}.png"
    plt.savefig(fn,dpi=300,transparent=True,bbox_inches='tight')
    print(f"Saved Image to [{fn}]")
    plt.clf()
    plt.cla()
    plt.close("all")
        

def experiment_global_optima_quality(cfg,records,exp_fields,exp_int):

    # -- extract & print summary info --
    record = records[exp_int]
    for exp_field in exp_fields:
        if exp_field == 'bss': continue
        # print(exp_field,record[exp_field])
    bss = record['bss']
    bss_ibatch = record['bss_ibatch']
    nblocks = record['nblocks']
    optima_index = find_optima_index(bss,bss_ibatch,nblocks)
    # nodynamic_index,optima_nd_index = find_nodynamic_optima_idx(bss,bss_ibatch,nblocks)

    # -- compute optima quality for each pixel function --
    record = records[exp_int]
    quality = edict()
    print(optima_index)
    assess_pixel_fields(quality,record,optima_index,nblocks,bss,bss_ibatch)
    # assess_flownet_fields(quality,record,optima_index,nblocks,bss,bss_ibatch)
    return quality

def assess_flownet_fields(quality,record,optima_index,nblocks,bss,bss_ibatch):
    fields = get_flownet_fields(record)
    gt_flow = torch.LongTensor(rearrange(record['gt_flow'],'tm1 b p -> b tm1 p'))
    ps,P = record['patchsize'],record['npatches']
    s,e = 0,ps*P
    for field in fields:
        quality[field] = edict({'correct':0,'total':0,'acc':0})

        # -- flow fields --
        flow = record[field]['flows']
        flow = tvF.crop(flow,s,s,e,e)
        flow = rearrange(record[field]['flows'],'tm1 b p h w -> b tm1 p (h w)')
        flow = torch.mean(flow,dim=3)
        flow_fp = flow.clone()
        flow = torch.round(flow) # round to ints

        eq = gt_flow == flow
        perc_zero = torch.mean((flow == 0).float()).item() * 100
        print("Precent 0: {:.2f}".format(perc_zero))
        if record['patchsize'] == 3 and record['noise_type'] == "g-1p0":
            # print(eq)
            inspect = 7
            # print(eq)
            print(eq[inspect])
            print(gt_flow[inspect])
            print(flow[inspect])
            print(flow_fp[inspect])
            print(record['patchsize'],record['noise_type'])
            # exit()
            
        # -- accuracy across time --
        # flow_error = torch.sum(torch.abs(flow),dim=2)
        correct = torch.all(torch.all(flow == gt_flow,dim=2).float(),dim=1)
        correct_list = list(correct.numpy())

        # -- compute quality scores --
        quality[field].correct = np.sum(correct_list)
        quality[field].total = len(correct_list)
        quality[field].acc = 100*quality[field].correct/quality[field].total
        quality[field].correct_v = '.'.join([str(x) for x in correct_list])

        print(flow.shape)
    return quality
        
def assess_pixel_fields(quality,record,optima_index,nblocks,bss,bss_ibatch):
    fields = get_pixel_fields(record)
    for field in fields:
        quality[field] = edict({'correct':0,'total':0,'acc':0})
        scores = rearrange(record[field]['scores'],'p ib ss -> ss ib p')
        scores_t = rearrange(record[field]['scores_t'],'p ib ss t -> ss t ib p')
        batch_size = record['batch_size']
        SS,IB,P = scores.shape
        # search_space batch, image batch, npatches
        scores = torch.mean(scores,dim=2) # ave over image patches
        
        # -- find optimal score index --
        argmins = torch.argmin(scores,0)
        print(field,argmins)

        # -- compute quality --
        correct_list = list((optima_index == argmins).numpy().astype(int))
        quality[field].correct = np.sum(correct_list)
        quality[field].total = len(optima_index)
        quality[field].acc = 100*quality[field].correct/quality[field].total
        quality[field].correct_v = '.'.join([str(x) for x in correct_list])

        # # -- filter bss to only almost no dynamics --
        # scores = scores[nodynamic_index]
        # argmins = torch.argmin(scores,0)

        # # -- compute quality --
        # quality[field].nd_correct = torch.sum(optima_nd_index == argmins).item()
        # quality[field].nd_total = len(optima_index)
        # quality[field].nd_acc = 100*quality[field].nd_correct/quality[field].nd_total

    return quality


        
