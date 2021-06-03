"""
Local patch alignment search

"""

# -- python imports --
import copy
import numpy as np
import numpy.random as npr
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch imports --
import torch
import torch.multiprocessing as mp
import torchvision.transforms.functional as tvF
cc = tvF.center_crop

# -- project imports --

# -- [local] project imports --
from pyutils import align_burst_from_flow,align_burst_from_block,global_flow_to_blocks,global_blocks_to_flow,print_tensor_stats,create_meshgrid,tile_across_blocks,global_blocks_ref_to_frames,global_flow_frame_blocks
from .utils import get_sobel_patches,save_image,get_ref_block_index
from .optimizer import AlignmentOptimizer

def lpas_spoof(burst,motion,nblocks,mtype,acc):
    T = burst.shape[0]
    ref_block = get_ref_block_index(nblocks)
    gt_blocks = global_flow_frame_blocks(motion,nblocks)
    # gt_blocks = global_flow_to_blocks(motion,nblocks) -- this returns frame alignment
    rands = npr.uniform(0,1,size=motion.shape[0])
    scores,blocks = [],[]
    for idx,rand in enumerate(rands):
        if rand > acc:
            fake = torch.randint(0,nblocks**2,(T,))
            fake[T//2] = ref_block
            blocks.append(fake)
        else: blocks.append(gt_blocks[idx])
        scores.append(0)
    blocks = torch.stack(blocks)
    burst_clone = burst.clone()
    aligned = align_burst_from_block(burst_clone,blocks,nblocks,"global")
    # print_tensor_stats("[lpas]: burst0 - burst1",burst[0] - burst[1])
    # print_tensor_stats("[lpas]: aligned0 - aligned1",aligned[0] - aligned[1])
    # print_tensor_stats("[lpas]: aligned[T/2] - burst[T/2]",aligned[T//2] - burst[T//2])
    return scores,aligned

def lpas_search(burst,ref_frame,nblocks,motion=None,method="simple",
                to_align=None,noise_info=None):
    """
    cfg: edict with parameters
    burst: shape: T,B,C,H,W (T = number of frames, B = batch)
    """
    # if not(motion is None):
    # print(motion)
    # gt_motion = global_flow_to_blocks(motion,nblocks)
    gt_motion = global_flow_frame_blocks(motion,nblocks)
    # print(gt_motion)
    # gt_motion = global_blocks_ref_to_frames(gt_motion,nblocks)
    # gt_motion = global_blocks_ref_to_frames(gt_motion,nblocks)
    # # flow = global_blocks_to_flow(gt_motion,nblocks)
    # print(gt_motion)

    # -- test --
    # T = burst.shape[0]
    # burst_clone = burst.clone()
    # aligned = align_burst_from_block(burst_clone,gt_motion,nblocks,"global")
    # save_image(cc(aligned - burst[T//2],96),"aligned_mid.png")
    # print_tensor_stats("[lpas]: aligned - burst[T//2]",cc(aligned - burst[T//2],96))
    # print_tensor_stats("[lpas]: burst0 - burst1",burst[1] - burst[2])
    # save_image(aligned[:-1] - aligned[1:],"aligned_offsetT.png")
    # print_tensor_stats("[lpas]: burst0 - burst1",cc(burst[0] - burst[1],96))
    # print_tensor_stats("[lpas]: aligned0 - aligned2",cc(aligned[0] - aligned[2],96))
    # print_tensor_stats("[lpas]: aligned0 - aligned1",cc(aligned[0] - aligned[1],96))
    # print_tensor_stats("[lpas]: aligned1 - aligned2",cc(aligned[1] - aligned[2],96))
    # print_tensor_stats("[lpas]: aligned2 - aligned3",cc(aligned[2] - aligned[3],96))
    # exit()

    # other = torch.LongTensor([[10, 11, 12, 13],
    #                           [21, 17, 12,  8],
    #                           [23, 18, 12,  7],
    #                           [ 9,  8, 12, 11]])
    # print(other)
    # aligned = align_burst_from_block(burst_clone,other,nblocks,"global")
    # # # print_tensor_stats("[lpas]: burst0 - burst1",burst[1] - burst[2])
    # # save_image(aligned[:-1] - aligned[1:],"aligned_offsetT.png")
    # # print_tensor_stats("[lpas]: burst0 - burst1",cc(burst[0] - burst[1],96))
    # print_tensor_stats("[lpas]: aligned0 - aligned2",cc(aligned[0] - aligned[2],96))
    # print_tensor_stats("[lpas]: aligned0 - aligned1",cc(aligned[0] - aligned[1],96))
    # print_tensor_stats("[lpas]: aligned1 - aligned2",cc(aligned[1] - aligned[2],96))
    # print_tensor_stats("[lpas]: aligned2 - aligned3",cc(aligned[2] - aligned[3],96))

    # # dmid = cc(aligned[T//2] - burst[T//2],96)
    # # print_tensor_stats("[lpas]: aligned[T/2] - burst[T/2]",dmid)

    # T,B = burst.shape[:2]
    # nframes = T
    # tiled = tile_across_blocks(burst_clone[[T//2]],nblocks)
    # rep_burst = repeat(burst,'t b c h w -> t b g c h w',g=tiled.shape[2])
    # for t in range(nframes):
    #     G = tiled.shape[2]
    #     delta_tg = tiled[0] - rep_burst[t]
    #     save_image(delta_tg,f"tiled_sub_burst_{t}.png")
    #     # for g in range(G):
    #     #     delta_tg = tiled[0,:,g] - rep_burst[t,:,g]
    #     #     for b in range(B): print_tensor_stats(f"[{t}_{g}_{b}]",delta_tg[b])

    # B = burst.shape[1]
    # for b in range(B):
    #     block_b = other[b]
    #     for t in range(T):
    #         index = block_b[t]
    #         print_tensor_stats(f"o:[{b}_{t}]",tiled[0,b,index]-burst_clone[nframes//2])
    # for b in range(B):
    #     block_b = gt_motion[b]
    #     for t in range(T):
    #         index = block_b[t]
    #         print_tensor_stats(f"a:[{b}_{t}]",tiled[0,b,index]-burst_clone[nframes//2])
    # exit()

    # -- def vars + create patches --
    nframes = burst.shape[0]
    T,B,C,H,W = burst.shape 
    num_patches,patchsize = 2,96
    patches,locations = get_sobel_patches(burst,nblocks,num_patches,patchsize)
    B,R,T,H,C,PS,PS = patches.shape


    # -- create helper objects --
    nsteps = 5
    nparticles = 100
    isize = burst.shape[-1]**2
    # score_params = edict({'stype':'raw','name':'smsubset'})
    # score_params = edict({'stype':'raw','name':'lgsubset'})
    # score_params = edict({'stype':'raw','name':'lgsubset_v_indices'})
    # score_params = edict({'stype':'raw','name':'extrema'})
    score_params = edict({'stype':'raw','name':'ave'})
    optim = AlignmentOptimizer(nframes,nblocks,ref_frame,isize,'global_const',
                               nsteps,nparticles,motion,score_params,noise_info)
    

    # -- compare with testing score --
    # optim.verbose = True
    # optim.init_samples()
    # gt_scores,gt_blocks = optim.sample(patches,gt_motion,K=1)
    # optim.verbose = False
    # print("-"*30)
    # print(gt_scores)
    # print(gt_blocks)

    # print("-"+"=-"*30)
    # optim.init_samples()
    # others = torch.LongTensor([[13, 13, 12, 11],[18, 17, 12,  6],
    #                   [17, 16, 12,  7],[11,  6, 12, 13]])
    # gt_scores,gt_blocks = optim.sample(patches,others,K=1)
    # print("-"*30)
    # print(gt_scores)
    # print(gt_blocks)

    # aligned_o = align_burst_from_block(burst,others,nblocks,"global")
    # aligned_gt = align_burst_from_block(burst,gt_motion,nblocks,"global")
    # save_image(aligned_o - burst[ref_frame],"aligned_o.png")
    # save_image(aligned_o - aligned_o[ref_frame],"aligned_o_i.png")
    # save_image(aligned_gt - burst[ref_frame],"aligned_gt.png")
    # save_image(aligned_gt - aligned_gt[ref_frame],"aligned_gt_i.png")

    # T,REF_H,B = optim.nframes,optim.get_ref_h(),patches.shape[1]
    # tgrid = torch.arange(patches.shape[2])
    # ref_patch = patches[0,0,T//2,REF_H]
    # save_image(patches[0,0,tgrid,others[0]] - ref_patch,"patches_o.png")
    # save_image(patches[0,0,tgrid,gt_motion[0]] - ref_patch,"patches_gt.png")
    # exit()
    optim.init_samples()

    
    # -- execute a simple search (one with many issues and few features) --
    verbose = False
    if verbose: print(f"Method: [{method}]")
    if method == "simple":
        simple_search(optim,patches)
    elif method == "exhaustive":
        exhaustive_search(optim,patches)
    elif method == "split":
        optim.score_params['name'] = 'ave'
        split_search(optim,patches)
    else:
        raise ValueError(f"Uknown search method [{method}]")

    # -- return best ones from subsamples --
    scores,blocks = optim.get_best_samples()
    
    # -- optimize over the final sets --
    score,block = scores[:,0],blocks[:,0]
    # print(score)
    # print(block)
    # print("-"*30)
    if verbose: print(block)

    # -- dynamic error --
    dynamic_acc = 0
    for b in range(B):
        dynamic_acc += torch.mean((gt_motion[b].cpu() == block[b].cpu()).float())
    # print(score)
    dynamic_acc = dynamic_acc/B

    # -- recover aligned burst images --
    # print(block)
    # gt_motion = global_blocks_ref_to_frames(block,nblocks)
    # print("Block")
    # print(block)
    # frame_block = global_blocks_ref_to_frames(block,nblocks)
    # print("Frame Block")
    # print(frame_block)
    # print(block)
    if to_align is None: to_align = burst
    aligned = align_burst_from_block(to_align,block,nblocks,"global")

    return score,aligned,dynamic_acc # indices for each neighborhood


def exhaustive_search(optim,patches):

    # -- initialize fized frames --
    T,REF_H,B = optim.nframes,optim.get_ref_h(),patches.shape[0]
    fixed_frames = edict()
    # -- no motion --
    ref_grid = torch.LongTensor([REF_H for t in range(T)])
    fixed_frames = edict()
    fixed_frames.idx = [optim.ref_frame]
    fixed_frames.vals = [REF_H]
    block_grids = optim.block_sampler.sample(None,fixed_frames,None)
    scores,blocks = optim.sample(patches,block_grids,K=3)
    return scores,blocks

def split_search(optim,patches):

    # -- init states of optimizer --
    motion = optim.motion_sampler.init_sample()
    fixed_frames = optim.frame_sampler.init_sample()
    block_grids = optim.block_sampler.init_sample()
    scores,blocks = optim.sample(patches,block_grids,K=10)

    # -- initialize fized frames --
    T,REF_H,B = optim.nframes,optim.get_ref_h(),patches.shape[0]
    fixed_frames = edict()

    # -- no motion --
    ref_grid = torch.LongTensor([REF_H for t in range(T)])
    fixed_frames = []
    for b in range(B):
        fixed_frames_b = edict()
        fixed_frames_b.idx = [t for t in range(T)]
        fixed_frames_b.vals = [REF_H for t in range(T)]
        fixed_frames.append(fixed_frames_b)

    # -- random --
    optim.nsteps = 1
    K_sched = [1]

    # -- search --
    for i in range(optim.nsteps):
        K = K_sched[i]

        """
        1. init samples
        2. seach each frame separately
        3. keep top k
        4. search meshgrid of k^(t-1)
        5. pick minima
        """

        # -- search along each separate frame --
        parallel = False
        args = (optim,patches,ref_grid,motion,B,K)
        blocks_i = execute_split_frame_search(T,fixed_frames,parallel,*args)

        # -- complete procs and unpack results --
        block_grids = create_mesh_frame_grid(blocks_i,B,T)

        # -- eval over grid --
        optim.init_samples()
        scores,blocks = optim.sample(patches,block_grids,K=K)

        # -- pick best optima --
        scores,blocks = optim.get_best_samples(K=1)

def execute_split_frame_search(T,fixed_frames,parallel,*args):
    procs,proc_limit = [],10
    if parallel: blocks_i = mp.Manager().dict()
    else: blocks_i = {}
    for t in range(T):
        fixed_frames_t = copy.deepcopy(fixed_frames)
        if parallel:
            p = mp.Process(target=search_across_frame,
                           args=(t,blocks_i,fixed_frames_t,*args))
            p.start()
            procs.append(p)
            # -- wait and reset proc queue --
            if len(procs) == proc_limit:
                finish_procs(procs,proc_limit)
                procs = []
        else:
            search_across_frame(t,blocks_i,fixed_frames_t,*args)
    finish_procs(procs,proc_limit)
    blocks_i = [blocks_i[str(t)] for t in range(T)]
    # if parallel: blocks_i = copy.deepcopy(blocks_i)
    return blocks_i

def finish_procs(procs,proc_limit):
    for p in procs: p.join()
    
def search_across_frame(t,blocks_i,fixed_frames,optim,patches,ref_grid,motion,B,K):
    # print("Search Across Frame",t,flush=True)
    if t == optim.ref_frame:
        mid = ref_grid[[t]].repeat(B,1)
        blocks_i[str(t)] = mid#.append(mid)
        return

    # -- select frames --
    fixed_frames_t = copy.deepcopy(fixed_frames)
    for b in range(B):
        del fixed_frames_t[b].idx[t]
        del fixed_frames_t[b].vals[t]

    # -- create grid --
    block_grids = optim.block_sampler.sample(None,fixed_frames_t,motion)

    # -- eval over grid --
    scores,blocks = optim.sample(patches,block_grids,K=K)
    topk_index_t = blocks[:,:,t]
    blocks_i[str(t)] = topk_index_t #.append(topk_index_t)

def simple_search(optim,patches):

    # -- init states of optimizer --
    motion = optim.motion_sampler.init_sample()
    fixed_frames = optim.frame_sampler.init_sample()
    block_grids = optim.block_sampler.init_sample()
    scores,blocks = optim.sample(patches,block_grids,K=10)

    # -- initialize fized frames --
    T,REF_H,B = optim.nframes,optim.get_ref_h(),patches.shape[0]
    fixed_frames = edict()

    # -- no motion --
    ref_grid = torch.LongTensor([REF_H for t in range(T)])
    # fixed_frames = edict()
    # fixed_frames.idx = [t for t in range(T)]
    # fixed_frames.vals = [REF_H for t in range(T)]
    fixed_frames = []
    for b in range(B):
        fixed_frames_b = edict()
        fixed_frames_b.idx = [t for t in range(T)]
        fixed_frames_b.vals = [REF_H for t in range(T)]
        fixed_frames.append(fixed_frames_b)

    # -- random --
    # fixed_frames.vals = list(torch.randint(0,optim.nblocks**2,(T,)).numpy())
    # fixed_frames.vals[T//2] = REF_H
    optim.nsteps = 4
    K_sched = [1,]*optim.nsteps
    K_sched[1] = 3
    K_sched[2] = 2

    # -- search --
    for i in range(optim.nsteps):
        K = K_sched[i]

        """
        1. init samples
        2. seach each frame separately
        3. keep top k
        4. search meshgrid of k^(t-1)
        5. pick minima
        """

        # -- search along each separate frame --
        parallel = False
        args = (optim,patches,ref_grid,motion,B,K)
        blocks_i = execute_split_frame_search(T,fixed_frames,parallel,*args)
        
        # -- create grid from frame optima --
        # "blocks_i.shape" = [ T, B, K ]
        # blocks_i[t] is a search along frame t
        # blocks_i[t][b,k] is the arangement for batch b with rank k
        # a mesh converts topk from each batch across outer "T"?
        # "create_mesh_frame_grid"
        # print(len(blocks_i))
        # print(blocks_i[0].shape)
        # print(blocks_i)
        block_grids = create_mesh_frame_grid(blocks_i,B,T)
        # print(len(block_grids))
        # print(block_grids[0].shape)
        # block_grids = rearrange(block_grids,'b r t -> (b r) t')
        # print(block_grids)
        # print(block_grids[0])
        # print(block_grids[0].shape)

        # -- eval over grid --
        optim.init_samples()
        scores,blocks = optim.sample(patches,block_grids,K=K)

        # -- pick best optima --
        scores,blocks = optim.get_best_samples(K=1)

        # -- re-init fixed frames for another search --
        # fixed_frames.idx = [t for t in range(T)]
        # for t in range(T):
        #     vals_t = []
        #     fixed_frames.vals[t] = np.unique(list(blocks[:,0,t].cpu().numpy()))
        for b in range(B):
            fixed_frames[b].idx = [t for t in range(T)]
            fixed_frames[b].vals = list(blocks[b,0,:].cpu().numpy())
        # print(K,fixed_frames)
        # print("-"*30)
        # search_frames = optim.frame_sampler.sample(scores,motion)


def create_mesh_frame_grid(blocks_i,B,T):
    """
    Compute each batch separately

    Take top K results from each individual (T-1) frame searches
    and compute the associated meshgrid.

    We will eval over entire grid and return optima as new starting point.
    """

    mesh = []
    for b in range(B):
        lists = []
        for t in range(T):
            list_t = list(blocks_i[t][b].cpu().numpy())
            lists.append(list_t)
        mesh_b = create_meshgrid(lists)
        mesh.append(mesh_b)
    mesh = torch.LongTensor(mesh)
    mesh = rearrange(mesh,'b r t -> r b t')
    return mesh
