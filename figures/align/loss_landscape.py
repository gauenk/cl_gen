
"""
Compare L2 with Boostrapping for 3 frames.

"""

# -- setup paths --
import sys,os
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib/")
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/experiments/noisy_burst_pixel_alignment/")

# -- python imports --
import numpy as np
import pandas as pd
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- torch imports --
import torch
import torch.nn.functional as F

# -- python plotting imports --
import matplotlib
matplotlib.use("agg")
matplotlib.rcParams['text.usetex'] = True
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# -- alignment imports --
from pyutils import tile_patches,save_image,torch_to_numpy
from patch_search import get_score_function
from align.combo.eval_scores import EvalBlockScores
from align.xforms import align_from_pix,flow_to_pix,create_isize,pix_to_flow,align_from_flow,flow_to_blocks
from align.combo.optim.v3._utils import init_optim_block,exh_block_range,get_ref_block
from align.combo.optim.v3._blocks import get_search_blocks

# -- project imports --
from pyplots.legend import add_legend
from pyutils import print_tensor_stats
from align.xforms import flow_to_pix,align_from_flow
from datasets.wrap_image_data import load_image_dataset,sample_to_cuda
from unsup_denoising.experiments.compare_to_competitors._aligned_methods import get_align_method

# -- local imports --
from configs import get_cfg_defaults

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def vprint(*args,**kwargs):
    verbose = True
    if verbose:
        print(*args,**kwargs)

def plot_landscape(scores,blocks,seed):

    # -- plot settings --
    colors = {'noisy':{'ave':'yellow','bs':'blue'},
              'clean':{'ave':'red','bs':'black'}}

    # -- pix index --
    pindex = 16*32+16
    pindex_list = [10*32+10,8*32+16,16*32+16,20*32+14]
    methods = ['ave','bs']
    for pindex in pindex_list:
    
        # -- get max_y --
        max_y = 0
        for itype in scores.keys():
            for pm_method in methods:#scores[itype].keys():
                max_y_tmp = np.max(scores[itype][pm_method][pindex])
                if max_y_tmp > max_y: max_y = max_y_tmp
    
        # -- create figs --
        fig,ax = plt.subplots(1,1,figsize=(8,4))
        ax = [ax]
        handles = []
        for itype in scores.keys():
            for pm_method in methods:#scores[itype].keys():
                # print(blocks[itype][pm_method].shape)
                # print(scores[itype][pm_method].shape)
                xgrid = blocks[itype][pm_method][pindex]
                ygrid = scores[itype][pm_method][pindex]
    
                order = np.argsort(xgrid)
                xgrid = xgrid[order]
                ygrid = ygrid[order]
    
                argmin_y = np.argmin(ygrid)
                min_x = xgrid[argmin_y]
    
                label = f"[{itype}]: {pm_method}"
                noise = np.random.normal(loc=0,scale=0.1,size=(1,))
                ax[0].vlines(min_x+noise,0,max_y,color=colors[itype][pm_method],linewidth=2)
                h = ax[0].plot(xgrid,ygrid,'-x',label=label,color=colors[itype][pm_method])
                handles.append(h)
    
        ax[0].set_xlabel("Index of Proposed Patch in Search Space.",fontsize=15)
        ax[0].set_ylabel("Measure of Alignment",fontsize=15)
    
        # -- add legend --
        # titles = []
        # handles = []
        # for field in fields:
        #     sfield = field.replace("_"," ").title()
        #     titles.append(sfield)
        #     pop = patches.Patch(color=colors[field], label=sfield)
        #     handles.append(pop)
        # ax[-1].axis("off")
        # box = ax[-2].get_position()
        # ax[-1].set_position([box.x0, box.y0,
        #                      box.width, box.height])
        # add_legend(ax[-1],"Methods",titles,handles,shrink = False,fontsize=12,
        #            framealpha=0.0,ncol=1)
    
        # -- save plot --
        plt.savefig(f"./loss_landscape_{seed}_{pindex}.png",
                    transparent=True,dpi=300,bbox_inches='tight')
        plt.close("all")
        plt.clf()
    
def search_blocks_to_str(search_block):
    search_block_str = []
    for elem in search_block:
        elem_str = '-'.join([str(e) for e in list(elem)])
        search_block_str.append(elem_str)
    return search_block_str

def search_blocks_to_labels(search_block,search_cats):
    search = search_blocks_to_str(search_block)
    cats = pd.Categorical(search, categories=search_cats)
    labels,_ = pd.factorize(cats,sort=True)
    return labels

def batch_search_blocks_to_labels(batch_blocks,block_strings):
    npix = batch_blocks.shape[0]
    labels = []
    for p in range(npix):
        label = search_blocks_to_labels(batch_blocks[p],block_strings)
        labels.append(label)
    labels = np.stack(labels)
    return labels

def boxes_from_flow(flow,h,w):
    isize = edict({'h':h,'w':w})
    flow_rs = rearrange(flow,'b (h w) t two -> t b h w two',h=h,w=w)
    pix = flow_to_pix(flow,isize=isize)
    pix = rearrange(pix,'b (h w) t two -> t b h w two',h=h,w=w)
    return pix

def tile_pair_to_full(scores,blocks,srch_blocks,
                      frames_pair,nframes,nblocks):
    """

    Paste the "scores" from "blocks" 
    to the associated "srch_blocks"

    """

    # -- find indices of srch_blocks_pair in srch_blocks --
    print(blocks.shape)
    print(srch_blocks.shape)
    npix,naligns_pair,two = blocks.shape
    npix,naligns,nframes = srch_blocks.shape
    scores_full = np.zeros((npix,naligns))
    assert len(frames_pair) == 1,"pair only uses a single, non-ref frame"
    sel_blocks_full = srch_blocks[:,:,frames_pair] # all blocks of this frame
    sel_blocks_pair = blocks[:,:,frames_pair]
    for bvalue in range(nblocks**2): # or range(naligns_pair)
        args_full = np.where(bvalue == sel_blocks_full)
        args_full_x,args_full_y = args_full[0],args_full[1]

        args_pair = np.where(bvalue == sel_blocks_pair)
        args_pair_x,args_pair_y = args_pair[0],args_pair[1]
        counts = np.bincount(args_pair_x)

        assert np.all(counts == 1),"Only one occurance for pair blocks."

        for p in range(npix):
            scores_full[p,args_full_y] = scores[p,args_pair_y[p]]

    return scores_full,srch_blocks

def run_with_seed(seed):
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #             Settings
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- settings --
    cfg = get_cfg_defaults()
    cfg.use_anscombe = False
    cfg.noise_params.ntype = 'g'
    cfg.noise_params.g.std = 25.
    cfg.nframes = 3
    cfg.dynamic_info.nframes = cfg.nframes
    cfg.nblocks = 3
    cfg.patchsize = 11
    cfg.gpuid = 1
    cfg.device = f"cuda:{cfg.gpuid}"

    # -- seeds --
    cfg.seed = seed
    # cfg.seed = 123 # sky of a forest 
    # cfg.seed = 345 # handrail and stairs
    # cfg.seed = 567 # cloudy blue sky
    # cfg.seed = 567 # cloudy blue sky

    # -- set seed --
    set_seed(cfg.seed)

    # -- load dataset --
    data,loaders = load_image_dataset(cfg)
    train_iter = iter(loaders.tr)

    # -- fetch sample --
    sample = next(train_iter)
    sample_to_cuda(sample)

    # -- unpack data --
    noisy,clean = sample['noisy'],sample['burst']
    nframes,nimages,ncolors,H,W = noisy.shape
    isize = edict({'h':H,'w':W})
    
    # -- setup results --
    scores = edict()
    scores.noisy = edict()
    scores.clean = edict()
    blocks = edict()
    blocks.noisy = edict()
    blocks.clean = edict()


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #        Setup For Searches    
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- tile image to patches --
    pad = 2*(cfg.nblocks//2)
    h,w = cfg.patchsize+pad,cfg.patchsize+pad

    noisy_patches = tile_patches(noisy,cfg.patchsize+pad).pix
    noisy_patches = rearrange(noisy_patches,'b t s (h w c) -> b s t c h w',h=h,w=w)
    nimages,npix,nframes,c,psH,psW = noisy_patches.shape

    clean_patches = tile_patches(clean,cfg.patchsize+pad).pix
    clean_patches = rearrange(clean_patches,'b t s (h w c) -> b s t c h w',h=h,w=w)
    nimages,npix,nframes,c,psH,psW = clean_patches.shape

    masks = torch.ones(nimages,npix,nframes,c,psH,psW).to(cfg.device)

    # -- create constants --
    frames = np.r_[np.arange(cfg.nframes//2),np.arange(cfg.nframes//2+1,cfg.nframes)]
    frames = repeat(frames,'z -> i s z',i=nimages,s=npix)
    brange = exh_block_range(nimages,npix,cfg.nframes,cfg.nblocks)
    curr_blocks = init_optim_block(nimages,npix,cfg.nframes,cfg.nblocks)
    srch_blocks = get_search_blocks(frames,brange,curr_blocks,cfg.device)    
    np_srch_blocks = torch_to_numpy(srch_blocks[0])
    S = len(srch_blocks[0,0])

    # -- create constants --
    frames_pair = np.array([0])
    frames = repeat(frames_pair,'z -> i s z',i=nimages,s=npix)
    brange = exh_block_range(nimages,npix,cfg.nframes,cfg.nblocks)
    curr_blocks_pair = init_optim_block(nimages,npix,cfg.nframes,cfg.nblocks)
    srch_blocks_pair = get_search_blocks(frames,brange,curr_blocks_pair,cfg.device)
    S_pair = len(srch_blocks[0,0])

    # -- encode blocks --
    single_search_block = srch_blocks[0,0].cpu().numpy()
    block_strings = search_blocks_to_str(single_search_block)
    labels = search_blocks_to_labels(single_search_block,block_strings)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #        Execute Searches
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


    #
    # --- run PAIRED split search ---
    #

    ave_fxn = get_score_function("ave")
    block_batchsize = 128
    evaluator = EvalBlockScores(ave_fxn,"ave",cfg.patchsize,block_batchsize,None)
    get_topK = evaluator.compute_topK_scores

    # -- a) run clean --
    clean_scores,clean_blocks = get_topK(clean_patches,masks,
                                         srch_blocks_pair,cfg.nblocks,S_pair)
    scores_full = torch_to_numpy(clean_scores[0])
    blocks_full = torch_to_numpy(clean_blocks[0])

    # -- b) tile results to full blocks --
    scores_full,blocks_full = tile_pair_to_full(scores_full,blocks_full,np_srch_blocks,
                                                frames_pair,cfg.nframes,cfg.nblocks)
    scores.clean.ave = scores_full
    blocks.clean.ave = batch_search_blocks_to_labels(blocks_full,block_strings)

    # -- a) run noisy --
    noisy_scores,noisy_blocks = get_topK(noisy_patches,masks,
                                         srch_blocks_pair,cfg.nblocks,S_pair)
    scores_full = torch_to_numpy(noisy_scores[0])
    blocks_full = torch_to_numpy(noisy_blocks[0])

    # -- b) tile results to full blocks --
    scores_full,blocks_full = tile_pair_to_full(scores_full,blocks_full,np_srch_blocks,
                                                frames_pair,cfg.nframes,cfg.nblocks)
    scores.noisy.ave = scores_full
    blocks.noisy.ave = batch_search_blocks_to_labels(blocks_full,block_strings)

    #
    # --- run FULL split search ---
    #

    ave_fxn = get_score_function("ave")
    block_batchsize = 128
    evaluator = EvalBlockScores(ave_fxn,"ave",cfg.patchsize,block_batchsize,None)
    get_topK = evaluator.compute_topK_scores

    # -- run clean --
    clean_scores,clean_blocks = get_topK(clean_patches,masks,srch_blocks,cfg.nblocks,S)

    clean_scores = torch_to_numpy(clean_scores)
    scores.clean.full_ave = clean_scores[0]

    clean_blocks = torch_to_numpy(clean_blocks)
    batch_blocks = clean_blocks[0,:,:,:]
    blocks.clean.full_ave = batch_search_blocks_to_labels(batch_blocks,block_strings)

    # -- run noisy --
    noisy_scores,noisy_blocks = get_topK(noisy_patches,masks,srch_blocks,cfg.nblocks,S)

    noisy_scores = torch_to_numpy(noisy_scores)
    scores.noisy.full_ave = noisy_scores[0]

    noisy_blocks = torch_to_numpy(noisy_blocks)
    batch_blocks = noisy_blocks[0,:,:,:]
    blocks.noisy.full_ave = batch_search_blocks_to_labels(batch_blocks,block_strings)


    #
    # --- run bootstrapping ---
    #

    bs_fxn = get_score_function("bootstrapping_mod2")
    block_batchsize = 32
    evaluator = EvalBlockScores(bs_fxn,"bs_mod2",cfg.patchsize,block_batchsize,None)
    get_topK = evaluator.compute_topK_scores

    # -- run noisy --
    noisy_scores,noisy_blocks = get_topK(noisy_patches,masks,srch_blocks,cfg.nblocks,S)

    noisy_scores = torch_to_numpy(noisy_scores)
    scores.noisy.bs = noisy_scores[0]

    noisy_blocks = torch_to_numpy(noisy_blocks)
    batch_blocks = noisy_blocks[0,:,:,:]
    blocks.noisy.bs = batch_search_blocks_to_labels(batch_blocks,block_strings)

    # -- run clean --
    clean_scores,clean_blocks = get_topK(clean_patches,masks,srch_blocks,cfg.nblocks,S)

    clean_scores = torch_to_numpy(clean_scores)
    scores.clean.bs = clean_scores[0]

    clean_blocks = torch_to_numpy(clean_blocks)
    batch_blocks = noisy_blocks[0,:,:,:]
    blocks.clean.bs = batch_search_blocks_to_labels(batch_blocks,block_strings)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #        Plot Results
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    print("Plotting Results.")
    plot_landscape(scores,blocks,seed)

def main():
    pid = os.getpid()
    print("Running loss_landscape.py")
    print(f"PID: {pid}")
    nrands = 20
    for r in range(nrands):
        random_seed = int(torch.rand(1)*100)
        print(f"random_seed: [{random_seed}]")
        run_with_seed(random_seed)

if __name__ == "__main__":
    main()

