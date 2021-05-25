

# -- python imports --
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.transforms as tvT

# -- project imports --
from settings import ROOT_PATH
from pyutils import images_to_psnrs
from datasets.transforms import get_noise_transform,get_dynamic_transform,get_noise_config
from explore.mesh import create_eval_mesh
from patch_search import get_score_function
from .vis import explore_record,plot_ave_nframes_scoreacc_noisetype
from .tile_utils import tile_burst_patches,aligned_burst_image_from_indices_global_dynamics
from .utils import get_ref_block_index,get_block_arangements

EVAL_DIR = Path(f"{ROOT_PATH}/output/lpas/eval_score")
if not EVAL_DIR.exists(): EVAL_DIR.mkdir(parents=True)

def eval_score(cfg,data,overwrite=False):

    eval_fn = Path(EVAL_DIR / "./default.csv")
    print(f"Eval filepath [{eval_fn}]")
    if (not eval_fn.exists()) or overwrite: record = run_eval_score(cfg,data,eval_fn)
    else: record = pd.read_csv(eval_fn)


    order = ['score_function','patchsize','noise_type','nframes','nblocks','ppf']
    explore_record(record)

    # plot_ave_nframes_scoreacc_noisetype(record)


def run_eval_score(cfg,data,eval_fn):

    exp_mesh,exp_fields = create_eval_mesh(cfg)
    record = init_record(exp_fields)
    align_clean_score = get_score_function("refcmp")

    # -- sample images --
    noise_xform,dynamic_xform,score_function = init_exp(cfg,exp_mesh[0])
    for image_id in tqdm(range(3)):

        # -- sample image --
        full_image = data.tr[image_id][2]

        # -- simulate dynamics --
        torch.manual_seed(123)
        burst = dynamic_xform(full_image)
        burst = burst.cuda(non_blocking=True)

        # -- tile clean --
        clean = tile_burst_patches(burst,cfg.patchsize,cfg.nblocks)

        # -- run over each experiment --
        for exp in tqdm(exp_mesh,leave=False):
            noise_xform,dynamic_xform,score_function = init_exp(cfg,exp)
            # block_search_space = get_block_arangements(exp.nblocks,exp.nframes)
            bss = get_small_test_block_arangements(EVAL_DIR,cfg.nblocks,cfg.nframes,2,3)
            block_search_space = bss 
            block_search_space.cuda(non_blocking=True)
    
            # -- sample noise --
            noisy = noise_xform(clean)
            
            # -- sample block collections --
            for block_id in tqdm(range(960,990,10),leave=False):
                # -- create blocks from patches and block index --
                clean_blocks = get_pixel_blocks(clean,block_id)
                noisy_blocks = get_pixel_blocks(noisy,block_id)

                # -- create filename to save loss landscape --
                score_paths = score_path_from_exp(eval_fn,exp,image_id,block_id)

                # -- compute scores for blocks --
                results = {}
                results["clean"] = alignment_optimizer(cfg,score_function,
                                                       clean_blocks,clean_blocks,
                                                       block_search_space,
                                                       score_paths.clean)
                results["noisy"] = alignment_optimizer(cfg,score_function,
                                                       noisy_blocks,clean_blocks,
                                                       block_search_space,
                                                       score_paths.noisy)
                results["align"] = alignment_optimizer(cfg,align_clean_score,
                                                       clean_blocks,clean_blocks,
                                                       block_search_space,
                                                       score_paths.align)
                results['dpixClean'] = compute_pixel_difference(clean_blocks,
                                                                     block_search_space)
                results['dpixNoisy'] = compute_pixel_difference(noisy_blocks,
                                                                     block_search_space)

            
                # -- append to record --
                record = update_record(record,exp,results,image_id,block_id)

    # -- save record --
    print(f"Saving record of information to [{eval_fn}]")
    record.to_csv(eval_fn)
    return record

def compute_pixel_difference(blocks,block_search_space):
    # -- vectorize search since single patch --
    R,B,T,N,C,PS1,PS2 = blocks.shape
    REF_N = get_ref_block_index(int(np.sqrt(N)))
    #print(cfg.nframes,T,cfg.nblocks,N,block_search_space.shape)
    assert (R == 1) and (B == 1), "single pixel's block and single sample please."
    expanded = blocks[:,:,np.arange(T),block_search_space]
    E = expanded.shape[2]
    R,B,E,T,C,H,W = expanded.shape
    PS = PS1

    ref = repeat(expanded[:,:,:,T//2],'r b e c h w -> r b e tile c h w',tile=T-1)
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta = F.mse_loss(ref[...,PS//2,PS//2],neighbors[...,PS//2,PS//2],reduction='none')
    delta = delta.view(R,B,E,-1)
    delta = torch.mean(delta,dim=3)
    pix_diff = delta[0,0]
    return pix_diff

def score_path_from_exp(eval_fn,exp,image_id,block_id):
    write_dir = eval_fn.parent
    exp_keys = sorted(list(exp.keys()))
    search_types = ['clean','noisy','align']
    paths = edict({})
    for search_type in search_types:
        exp_str = ''
        for key in exp_keys:
            exp_str += f'_{exp[key]}'
        stem = f'{search_type}{exp_str}' + f'{image_id}_{block_id}.npy'
        paths[search_type] = write_dir / stem
    return paths
    
    
def cog_evaluation(cfg,score_fxn,blocks,clean,block_search_space,scores_path):
    pass

def alignment_optimizer(cfg,score_fxn,blocks,clean,block_search_space,scores_path):

    # -- vectorize search since single patch --
    R,B,T,N,C,PS1,PS2 = blocks.shape
    REF_N = get_ref_block_index(int(np.sqrt(N)))
    #print(cfg.nframes,T,cfg.nblocks,N,block_search_space.shape)
    assert (R == 1) and (B == 1), "single pixel's block and single sample please."
    expanded = blocks[:,:,np.arange(T),block_search_space]
    E = expanded.shape[2]

    # -- evaluate block --
    scores = score_fxn(cfg,expanded)
    scores = scores[0,0]
    best_index = torch.argmin(scores).item()
    best_score = torch.min(scores).item()
    assert E >= best_index, "No score can be greater than best index."

    # -- select the best block --
    best_block = block_search_space[best_index]
    best_block_str = ''.join([str(i) for i in best_block.cpu().numpy()])

    # -- construct image and compute the associated psnr --
    ref = repeat(clean[0,0,T//2,REF_N],'c h w -> tile c h w',tile=T)
    aligned = clean[0,0,np.arange(T),best_block]
    psnr = images_to_psnrs(ref,aligned)
    
    # -- save scores to numpy array --
    np.save(scores_path,scores.cpu().numpy())

    # -- compute results --
    results = {'scores':scores_path,
               'best_idx':best_index,
               'best_score':best_score,
               'best_block':best_block_str,
               'psnr':psnr}
    return results
    
def get_pixel_blocks(patches,block_id):
    R,B,N,T,C,PS1,PS2 = patches.shape
    return patches[[block_id]]

    # # -- get blocks --
    # block_indices = np.arange(N)
    # frame_indices = np.arange(T)
    # neighbors = patches[:,:,block_indices,frame_indices]

    # # -- get ref --
    # ref_block_index = get_ref_block_index(int(np.sqrt(N)))
    # ref_frame_index = [T//2]
    # ref = patches[:,:,ref_block_index,ref_frame_index]
    
    # # -- combine --
    # blocks = torch.cat([ref,neighbors],dim=2)

    # return blocks

def init_record(exp_fields):
    record = edict()

    # -- init exp fields --
    for field in exp_fields: record[field] = []

    # -- init result fields --
    search_types = ['clean','noisy','align']
    result_fields = ['scores','best_idx','best_score']
    for search_type in search_types:
        for result_field in result_fields:
            record[f"{search_type}_{result_field}"] = []

    record = pd.DataFrame(record)
    return record

def update_record(record,exp,results,image_id,block_id):
    record_i = edict()

    # -- sample id --
    record_i['image_id'] = image_id
    record_i['block_id'] = block_id

    # -- exp fields --
    for field in exp.keys(): record_i[field] = exp[field]

    # -- record results --
    for search_type,result in results.items():
        for field in result.keys():
            record_i[f"{search_type}_{field}"] = result[field]

    # -- append to record --
    record = record.append(record_i,ignore_index=True)
    return record


def init_exp(cfg,exp):

    # -- set patchsize -- 
    cfg.patchsize = int(exp.patchsize)
    
    # -- set patchsize -- 
    cfg.nframes = int(exp.nframes)
    cfg.N = cfg.nframes

    # -- set number of blocks (old: neighborhood size) -- 
    cfg.nblocks = int(exp.nblocks)
    cfg.nh_size = cfg.nblocks # old name

    # -- get noise function --
    nconfig = get_noise_config(cfg,exp.noise_type)
    noise_xform = get_noise_transform(nconfig,use_to_tensor=False)
    
    # -- get dynamics function --
    cfg.dynamic.ppf = exp.ppf
    cfg.dynamic.bool = True
    cfg.dynamic.random_eraser = False
    cfg.dynamic.frame_size = cfg.frame_size
    cfg.dynamic.total_pixels = cfg.dynamic.ppf*(cfg.nframes-1)
    cfg.dynamic.frames = exp.nframes

    def nonoise(image): return image
    dynamic_info = cfg.dynamic
    dynamic_raw_xform = get_dynamic_transform(dynamic_info,nonoise)
    dynamic_xform = dynamic_wrapper(dynamic_raw_xform)

    # -- get score function --
    score_function = get_score_function(exp.score_function)

    return noise_xform,dynamic_xform,score_function

def dynamic_wrapper(dynamic_raw_xform):
    def wrapped(image):
        pil_image = tvT.ToPILImage()(image).convert("RGB")
        results = dynamic_raw_xform(pil_image)
        burst = results[0].unsqueeze(1)
        return burst
    return wrapped

