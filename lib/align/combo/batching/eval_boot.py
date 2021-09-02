
# -- python imports --
import time,nvtx,numba,tqdm,types
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
from pyutils import torch_to_numpy,save_image,images_to_psnrs
from pyutils.mesh_gen import BatchGen
import align.combo._block_utils as block_utils
from align._utils import BatchIter,burst_to_patches
from align.xforms import blocks_to_pix,flow_to_blocks

# -- weird deps we should remove --
from patch_search.pixel._indexing import index_along_frames

# -- local imports --
from ._template import EvalTemplate

class EvalBootBlockScores(EvalTemplate):

    def __init__(self,bs_fxn_limitB,bs_fxn,bs_fxn_name,patchsize,
                 block_batchsize,gpuid=1):
        super().__init__(bs_fxn,bs_fxn_name,patchsize,
                             block_batchsize,None,gpuid=1)
        self.limitB_fxn = bs_fxn_limitB

    def compute_scores(self,patches,blocks,state,nblocks,store_t=False):
        return self.compute_topK(patches,blocks,state,nblocks,-1,store_t)

    def compute_topK(self,patches,blocks,state,nblocks,K,store_t=False):
        if isinstance(blocks,BatchGen):
            return self.compute_topK_gen(patches,blocks,
                                         state,nblocks,K,store_t)
        else:
            return self.compute_topK_tensor(patches,blocks,
                                            state,nblocks,K,store_t)

    def exec_batch_full(self,batch,block_patches,patches,nblocks):

        # -- index search space  --
        batchsize = batch.shape[2]
        batch_patches = block_utils.index_block_batches_T(block_patches,patches,
                                                          batch,self.patchsize,
                                                          nblocks,self.gpuid)
        scores,scores_t = self.compute_batch_scores(batch_patches,None,None,None,"full")
        return scores,scores_t

    def exec_batch(self,batch,init_block,block_patches,prev_patches,patches,
                   prev_scores,nblocks):

        # -- get frames that changed from previous to current "blocks" --
        dframes = self._get_dframes(batch,init_block)

        # -- fill a batch of "block_patches" with correct offsets --
        batchsize = batch.shape[2]
        print("batch.shape ",batch.shape)
        print("init_block.shape ",init_block.shape)
        print("block_patches.shape ",block_patches.shape)
        print("prev_patches.shape ",prev_patches.shape)
        print("prev_scores.shape ",prev_scores.shape)
        print("patches.shape ",patches.shape)
        batch_patches = block_utils.index_patches_T(block_patches,patches,
                                                    batch,self.patchsize,
                                                    nblocks,self.gpuid)
        # nimages,npix,naligns,nframes,ncolors,psH,psW = batch_patches.shape

        # -- fill the previous patch "prev_patches" with correct offset --
        batch_prev_patches = block_utils.index_patches_T(prev_patches,
                                                         patches,init_block,
                                                         self.patchsize,
                                                         nblocks,self.gpuid)
        # nimages,npix,naligns_prev,nframes,ncolors,psH,psW = batch_prev_patches.shape

        # -- index only the frames that changing using "dframes" --
        print("batch_patches.shape ",batch_patches.shape)
        nimages = batch_patches.shape[0]
        shape_str = 'i p a t c h w -> i a p t (c h w)'
        bp_to_index = rearrange(batch_patches,shape_str)
        dframes_rs = rearrange(dframes,'i p a -> i a p')
        patches = index_along_frames(bp_to_index[0],dframes_rs[0])
        # print("patches.shape ",patches.shape)
        patches = rearrange(patches,'a p f -> p a f')[None,:]
        # patches = patches[None,:] # re-introduce the "nimages" dim again
        nimages,naligns,npatches,nftrs = patches.shape
        shape_str = 'i a p t f -> i p a t f'
        patches_full = rearrange(bp_to_index,shape_str)

        # -- previous patches to ftr vector --
        shape_str = 'i p a t c h w -> i p a t (c h w)'
        batch_prev_patches = rearrange(batch_prev_patches,shape_str)

        # -- compute directly for sanity check --
        scores,scores_t = self.compute_batch_scores(patches,batch_prev_patches,
                                                    prev_scores,dframes,patches_full,"step")
        return scores,scores_t

    def compute_batch_scores(self,patches,prev_patches,prev_scores,dframes,patches_full,ctype):
        if ctype == "full":
            return self.compute_batch_scores_full(patches)
        elif ctype == "step":
            return self.compute_batch_scores_step(patches,prev_patches,
                                                  prev_scores,dframes,patches_full)

    def compute_batch_scores_step(self,patches,prev_patches,prev_scores,dframes,patches_full):
        self.score_cfg.bs_type = "step"
        scores,scores_t = self.limitB_fxn(self.score_cfg,patches,prev_patches,
                                          prev_scores,dframes,patches_full)
        return scores,scores_t

    def compute_batch_scores_full(self,patches):

        # -- [new] each npix is computed separately --
        nimages = patches.shape[0]
        patches = rearrange(patches,'b p a t c h w -> 1 (b p) a t c h w')
        self.score_cfg.gpuid = self.gpuid # update with current gpuid
        self.score_cfg.bs_type = "full"
        scores,scores_t = self.score_fxn(self.score_cfg,patches)
        if not(scores_t is None):
            scores_t = rearrange(scores_t,'1 (b p) a t -> b p a t',b=nimages)
        scores = rearrange(scores,'1 (b p) a -> b p a',b=nimages)
        return scores,scores_t

    @nvtx.annotate("compute_topK_gen", color="orange")
    def compute_topK_gen(self,patches,blocks,state,nblocks,K,store_t=False):
        raise NotImplemented("")
        
    @nvtx.annotate("compute_topK_tensor", color="orange")
    def compute_topK_tensor(self,patches,blocks,state,nblocks,K,store_t=False):
        """
        Compute values of blocks using efficient bootstrapping method.
        Assumes all blocks are constrained to be only changed by "1" to the original.

        TODO: add check for block constraints

        Compute BS(S_i,blocks[0])

        Compute BS(S_i,blocks[1:]) = BS(S_i,blocks[0]) + Delta(S_i,blocks[0],blocks[1:])
        """

        # -----------------
        #   Init Function
        # -----------------

        # -- reset --
        self._clear()

        # -- current --
        nimages,nsegs,nframes = patches.shape[:3]
        pcolor,psH,psW = patches.shape[3:]
        mcolor = 1
        self.score_cfg.nframes = nframes

        # -- torch to numpy --
        device = patches.device
        self._update_gpuid(device)
        naligns = len(blocks[0][0])

        # -- setup batches --
        ps = self.patchsize
        batchsize = self.block_batchsize
        biter = BatchIter(naligns,batchsize)

        # -- get state information --
        state.blocks = state.blocks.to(device)
        nimages,nsegs,naligns,nframes = state.blocks.shape
        state_patches = np.zeros((nimages,nsegs,1,nframes,pcolor,ps,ps))
        state_patches = torch.FloatTensor(state_patches).to(device,non_blocking=True)
        state_patches = block_utils.index_block_batches(state_patches,patches,
                                                        state.blocks,
                                                        self.patchsize,
                                                        nblocks,self.gpuid)
        # -- allocate memory --
        block_patches = np.zeros((nimages,nsegs,batchsize,nframes,pcolor,ps,ps))
        block_patches = torch.FloatTensor(block_patches).to(device,non_blocking=True)

        block_prev_patches = np.zeros((nimages,nsegs,1,nframes,pcolor,ps,ps))
        block_prev_patches = torch.FloatTensor(block_prev_patches)
        block_prev_patches = block_prev_patches.to(device,non_blocking=True)

        block_masks = np.zeros((nimages,nsegs,batchsize,nframes,mcolor,ps,ps))
        block_masks = torch.FloatTensor(block_masks).to(device,non_blocking=True)
        nomotion = torch.LongTensor([4,]*nframes).reshape(1,nframes).to(device)
        nomo = True

        # ---------------------------------------------------
        #   [remove?] Evaluate Expensive Bootstrapping: One Time Only
        # ---------------------------------------------------
        # print("init_block.shape ",init_block.shape)
        # print("block_patches.shape ",block_patches.shape)
        # scores,scores_t = self.exec_batch_full(init_block,block_patches,
        #                                                  patches,nblocks)
        # block_utils.block_batch_update(self.samples,scores,
        #                                scores_t,init_block,K,store_t)
        # prev_scores,prev_scores_t = scores,scores_t
        init_block = state.blocks.to(device,non_blocking=True)
        prev_scores = state.scores.to(device,non_blocking=True)

        # ---------------------------------------------------
        #   Evaluate Bootstrapping Over Batch
        # ---------------------------------------------------
        
        for batch_indices in biter:

            print("blocks.shape ",blocks.shape)
            print("batch_indices ",batch_indices)
            batch = blocks[:,:,:,batch_indices].to(device)
            # batch,block_patches,block_prev_patches,block_prev_scores,
            # patches,prev_scores,nblocks
            scores,scores_t = self.exec_batch(batch,init_block,block_patches,
                                              block_prev_patches,patches,
                                              prev_scores,nblocks)
            block_utils.block_batch_update(self.samples,scores,
                                           scores_t,batch,K,store_t)
            torch.cuda.empty_cache()

        # if nomo is False: print("no motion detected.")
        scores,scores_t,blocks = block_utils.get_topK_samples(self.samples,K)

        # -- no gpu --
        scores = scores.cpu() # scores.shape = (nimages,nsegs,K)
        if torch.is_tensor(scores_t): scores_t = scores_t.cpu() # (nimages,nsegs,K,t)
        blocks = blocks.cpu() # blocks.shape = (nimages,nsegs,K,t)

        return scores,scores_t,blocks
        

    # ------------------------------------------------
    #
    #  Compute Scores from Images instead of Patches 
    #
    # ------------------------------------------------

    def score_burst_from_flow(self,burst,flow,init_block,state,patchsize,nblocks):
        # -- check shapes --
        nframes_burst = burst.shape[0]
        nimages,npix,naligns,nframes,two = flow.shape
        assert nframes == nframes_burst,"Num of frame must be equal."

        # -- compute score --
        if flow.ndim == 4:
            blocks = flow_to_blocks(flow,nblocks)
            blocks = rearrange(blocks,'i p t -> i p 1 t')
        elif flow.ndim == 5:
            flow = rearrange(flow,'i p a t two -> (i a) p t two')
            blocks = flow_to_blocks(flow,nblocks)
            blocks = rearrange(blocks,'(i a) p t -> i p a t',a=naligns)
        else:
            msg = f"Uknown ndims for flow input [{flow.ndim}]. We accept 4 or 5."
            raise ValueError(msg)
        return self.score_burst_from_blocks(burst,blocks,init_block,
                                            state,patchsize,nblocks)

    def score_burst_from_blocks(self,burst,blocks,state,patchsize,nblocks):

        # -- assert shape --
        assert blocks.ndim == 4,"Must have 4 dims: (nimages,npix,naligns,nframes)"
        nimages,npix,naligns,nframes = blocks.shape

        # -- get patches --
        pad = 2*(nblocks//2)
        patches = burst_to_patches(burst,patchsize+pad)

        # -- shapes --
        device = patches.device
        nimages,nsegs,nframes,pcolor,ps,ps = patches.shape
        mcolor,batchsize = 1,1

        # -- check patchsize and send warning --
        if self.patchsize != patchsize:
            print("patchsize for execution is not the same as object patchsize")
            print(f"Called with {patchsize} but created with {self.patchsize}")
        masks = np.zeros((nimages,npix,1,nframes,1,patchsize,patchsize))
        scores,scores_t,blocks = self.compute_scores(patches,blocks,
                                                     state,nblocks,store_t=True)

        # scores.shape = (nimages,nsegs,naligns)
        # scores_t.shape = (nimages,nsegs,naligns,nframes)
        # blocks.shape = (nimages,nsegs,naligns,nframes)
        print("[eval boot]: scores_t.shape", scores_t.shape)

        return scores,scores_t,blocks


    # --------------------
    #
    #    Misc Utilities
    #
    # --------------------

    @staticmethod
    def filter_blocks_to_1skip_neighbors(blocks,init_block):
        naligns,nframes = blocks.shape
        one,nframes = init_block.shape
        assert one == 1,"Only one initial alignment."
        diff = torch.abs(blocks - init_block)
        nchanges = torch.sum(diff,dim=1)
        adims = torch.where(nchanges == 1)[0]
        filtered = torch.index_select(blocks,0,adims)
        return filtered

    def _get_dframes(self,blocks,init_block):
        nimages,nsegs,naligns,nframes = blocks.shape
        nimages,nsegs,one,nframes = init_block.shape
        assert one == 1,"Only one initial alignment."
        diff = torch.abs(blocks - init_block)
        nchanges = torch.sum(diff,dim=3)
        assert torch.all(nchanges == 1), "Only one value can change from ref."
        dframes = torch.where(diff)[-1]
        dframes = rearrange(dframes,'(i s a) -> i s a',s=nsegs,a=naligns)
        return dframes
