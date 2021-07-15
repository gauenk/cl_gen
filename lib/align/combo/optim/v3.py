
# -- python imports --
import numpy as np
import numpy.random as npr
from einops import rearrange,repeat
from joblib import delayed

# -- pytorch imports --
import torch

# -- project imports --
from pyutils import create_meshgrid
from align.xforms import blocks_to_flow
from align._parallel import ProgressParallel
    
def run_image_batch(patches,masks,evaluator,
                    nblocks,iterations,
                    subsizes,K):
    
    PARALLEL = True
    if PARALLEL:
        return run_image_batch_parallel(patches,masks,evaluator,
                                        nblocks,iterations,
                                        subsizes,K)
    else:
        return run_image_batch_serial(patches,masks,evaluator,
                                      nblocks,iterations,
                                      subsizes,K)

def run_image_batch_parallel(patches,masks,evaluator,
                             nblocks,iterations,
                             subsizes,K):
    blocks = []
    nimages = patches.shape[0]
    pParallel = ProgressParallel(True,len(patches),n_jobs=8)
    delayed_fxn = delayed(run)
    blocks = pParallel(delayed_fxn(patches[[i]],masks[[i]],evaluator,
                                   nblocks,iterations,subsizes,K)
                       for i in range(nimages))
    blocks = torch.cat(blocks,dim=1) # B, T, R; R = H x W
    return blocks

def run_image_batch_serial(patches,masks,evaluator,
                           nblocks,iterations,
                           subsizes,K):
    blocks = []
    nimages = patches.shape[0]
    for b in range(nimages):
        blocks_b = run(patches[[b]],masks[[b]],evaluator,
                       nblocks,iterations,subsizes,K)        
        blocks.append(blocks_b)
    blocks = torch.cat(blocks) # B, T, R; R = H x W
    return blocks

def run(patches,masks,evaluator,
        nblocks,iterations,
        subsizes,K):

    # TODO: if patches are not rectangles, create masks
    # Current: assume patches are all rectangles
    # nftrs is flattened from ( C x max_seg_H x max_seg_W )
    # "mask" says where padding exists, ( C x max_seg_H x max_seg_W )
    # "locs" with shape (nimages,nframes,nsegs,nftrs,2) says where each ftr came from
    # actual features is flattened from ( C x max_seg_H x max_seg_W ) "and-ed" with a mask


    # -- init settings --
    nimages,nsegs,nframes = patches.shape[:3]
    assert nimages == 1,"Single batch per search."
    curr_blocks = init_optim_block(nimages,nsegs,nframes,nblocks)
    exh_brange = exh_block_range(nimages,nsegs,nframes,nblocks)

    for iter_i in range(iterations):

        # -- pick topK arrangements searching each frame separately --
        topK_blocks = split_frame_search(patches,masks,evaluator,curr_blocks,
                                         exh_brange,nblocks,K)
        
        # -- pick top arrangement search over rand subsets of frames --
        curr_blocks = rand_subset_search(patches,masks,evaluator,curr_blocks,
                                         topK_blocks,nblocks,subsizes)

    # return:
    # -- curr_blocks(torch.LongTensor) shape (nimages,nframes,nsegs) 
    #    -- each "long" represents the index from exhaustive search
    # -- "flow" (torch.LongTensor) shape (nimages,nframes,nsegs,2)
    #    -- each *pair of ints* represents the _nnf_ of each
    #       *segmentation* for each *frame*
    curr_blocks = rearrange(curr_blocks,'i s t -> (i s) t')
    print("[curr_blocks.shape]: ", curr_blocks.shape)
    flow = blocks_to_flow(curr_blocks,nblocks)
    print("[flow.shape]: ", flow.shape)
    flow = rearrange(flow,'(i s) t two -> t i s two',i=nimages)
    return flow

def get_ref_block(nblocks):
    return nblocks**2//2 + nblocks//2

def init_optim_block(nimages,nsegs,nframes,nblocks):
    ref_block = get_ref_block(nblocks)
    blocks = torch.ones((nimages,nsegs,nframes)).type(torch.long)
    blocks *= ref_block
    return blocks

def exh_block_range(nimages,nsegs,nframes,nblocks):
    full_range = torch.LongTensor([np.arange(nblocks**2) for t in range(nframes)])
    full_range = repeat(full_range,'t h -> b s t h',b=nimages,s=nsegs)
    full_range = full_range.type(torch.long)
    return full_range

def init_topK_split_search(nimages,nsegs,nframes,K):
    topk_blocks = torch.zeros((nimages,nsegs,nframes,K)).type(torch.long)
    return topk_blocks

def split_frame_search(patches,masks,evaluator,curr_blocks,brange,nblocks,K):
    r"""
    
    brange: a list of search ranges for each frame
    """
    # -- shapes and init --
    nimages,nsegs,nframes = patches.shape[:3]
    ones = torch.ones((nimages,nsegs,1))
    topk_blocks = init_topK_split_search(nimages,nsegs,nframes,K)
    assert nimages == 1,"Only batchsize 1 right now."
    ref_block = get_ref_block(nblocks)

    for t in range(nframes):
        if t == nframes//2:
            topk_blocks[:,:,t,:] = ref_block
            continue
        blocks_t = ones * t
        block_ranges = select_block_ranges(blocks_t,brange,curr_blocks)
        srch_blocks = mesh_blocks(block_ranges)
        srch_blocks = torch.LongTensor(srch_blocks)
        scores,blocks_t = evaluator.compute_topK_scores(patches,masks,srch_blocks,
                                                        nblocks,K)
        topk_blocks_t = blocks_t[:,:,:,t]
        topk_blocks[:,:,t] = topk_blocks_t

    return topk_blocks

def rand_subset_search(patches,masks,evaluator,curr_blocks,brange,nblocks,subsizes):
    r"""
    
    brange: a list of search ranges for each frame
    """
    
    # -- this would be a good place for particle filtering --
    # -- 1.) for each particle we would search along random subsets --
    # -- 2.) merge results periodically (e.g. take best one so far) --
    # -- 3.) alternate between (1) and (2)
    nimages,nsegs,nframes = patches.shape[:3]
    for size in subsizes:
        frames = repeat(npr.choice(nframes,size=size),'z -> i s z',i=nimages,s=nsegs)
        # frames = npr.choice(nframes,size=(nimages,nsegs,size))
        # frames = npr.permutation(nframes,size=(nimages,nsegs,size))
        block_ranges = select_block_ranges(frames,brange,curr_blocks)
        srch_blocks = mesh_blocks(block_ranges)
        srch_blocks = torch.LongTensor(srch_blocks)
        scores,blocks = evaluator.compute_topK_scores(patches,masks,srch_blocks,nblocks,1)
        curr_blocks = blocks[:,:,0,:]

    return curr_blocks

def select_block_ranges(frames,brange,curr_blocks):
    r"""
    frames: [ frame_index_1, frame_index_2, ..., frame_index_F ]
    brange[0,:,0]: [ [range_of_frame_1], [range_of_frame_2], ..., [range_of_frame_T] ]
    curr_blocks: [ block_1, block_2, ..., block_T ]

    frames.shape = (nimages,nsegs,M)
    brange.shape = (nimages,nsegs,nframes)

    create list of ranges (lists) for each frames for a meshgrid
    
    sranges[nested lists] shape = (nimages,nsegs,nframes,*)
    """
    nimages = len(brange)
    nsegs = len(brange[0])
    nframes = len(brange[0][0])

    sranges = []
    for b in range(nimages):
        sranges_b = []
        for s in range(nsegs):
            inputs = [frames[b][s],brange[b][s],curr_blocks[b][s]]
            srange_bs = select_block_ranges_bs(*inputs)
            sranges_b.append(srange_bs)
        sranges.append(sranges_b)
    return sranges

def select_block_ranges_bs(frames,brange,curr_blocks):
    srange,nframes = [],len(curr_blocks)
    for f in range(nframes):
        if f in frames:
            brange_u = np.unique(brange[f])
            selected_indices = np.atleast_1d(brange_u)
        else:
            selected_indices = np.atleast_1d(curr_blocks[f])
        srange.append(list(selected_indices))
    return srange

def mesh_blocks(brange):
    mesh = []
    nimages,nsegs = len(brange),len(brange[0])
    for b in range(nimages):
        mesh_b = []
        for s in range(nsegs):
            brange_bs = brange[b][s]
            grids = np.meshgrid(*brange_bs,indexing='ij')
            grids = [grids[g].flatten() for g in range(len(grids))]
            grids = rearrange(np.stack(grids),'t a -> a t')
            #grids = torch.LongTensor(grids)
            mesh_b.append(grids)
        mesh.append(mesh_b)
    # mesh = torch.LongTensor(mesh)
    # mesh = rearrange(mesh,'(b s) t a -> b s a t',b=nimages)
    return mesh

"""

(A) list of frames, single block, nblocks range for each frame
    - for a fixed block arangement, compute all permutations along a frame dim
(B) list of frames, a list of blocks, nblocks range for each frame
    - for a set of block arangements (one per frame)


K^S_n search over the top_K for each S_n frame randomly sampled

In (A) the range of each t is { 1, ..., nblocks }
In (B) the range of each t is { k_1, ..., k_K }

We need a fixed value for each frame who is not selected. ("optim_block")

We compute meshgrid over the selected frame's range

"""
