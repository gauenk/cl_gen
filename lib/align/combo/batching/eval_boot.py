
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

# -- local imports --
from ._template import EvalTemplate

class EvalBootBlockScores(EvalTemplate):

    def __init__(self,bs_fxn_limitB,bs_fxn,bs_fxn_name,patchsize,
                 block_batchsize,gpuid=1):
        super().__init__(bs_fxn,bs_fxn_name,patchsize,
                             block_batchsize,None,gpuid=1)
        self.limitB_fxn = bs_fxn_limitB

    def compute_scores(self,patches,masks,blocks,nblocks,store_t=False):
        return self.compute_topK(patches,masks,blocks,nblocks,-1,store_t)

    def compute_topK(self,patches,masks,blocks,nblocks,K,store_t=False):
        if isinstance(blocks,BatchGen):
            return self.compute_topK_gen(patches,masks,blocks,nblocks,K,store_t)
        else:
            return self.compute_topK_tensor(patches,masks,blocks,nblocks,K,store_t)

    def exec_batch(self,batch,block_patches,patches,tokeep,nblocks):

        # -- index search space  --
        batchsize = batch.shape[2]
        block_patches_i = block_utils.index_block_batches(block_patches,patches,
                                                          batch,tokeep,
                                                          self.patchsize,
                                                          nblocks,self.gpuid)
        # -- index patches along dframes --
        patches = index_along_frames(patches,dframes)
        prev_patches = index_along_frames(prev_patches,dframes)
        print("patches.shape", patches.shape)
        print("prev_patches.shape", prev_patches.shape)
        naligns,npatches,nftrs = patches.shape
        naligns,npatches_prev,nftrs = prev_patches.shape
        assert naligns_prev == 1, "Only one previous alignment is supported."
        

        # -- compute directly for sanity check --
        scores,scores_t = self.compute_batch_scores(block_patches_i,None)
        return scores,scores_t

    def compute_batch_scores(self,patches,prev_patches):

        # -- [new] each npix is computed separately --
        nimages = patches.shape[0]
        patches = rearrange(patches,'b p t a c h w -> 1 (b p) a t c h w')
        self.score_cfg.gpuid = self.gpuid # update with current gpuid
        scores,scores_t = self.score_fxn(self.score_cfg,patches,prev_patches)
        if not(scores_t is None):
            scores_t = rearrange(scores_t,'1 (b p) a t -> b p a t',b=nimages)
        scores = rearrange(scores,'1 (b p) a -> b p a',b=nimages)
        return scores,scores_t

    @nvtx.annotate("compute_topK_gen", color="orange")
    def compute_topK_gen(self,patches,masks,block_gens,nblocks,K,store_t=False):
        raise NotImplemented("")
        
    @nvtx.annotate("compute_topK_tensor", color="orange")
    def compute_topK_tensor(self,patches,blocks,scores_past,patches_past,
                            nblocks,K,store_t=False):
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
        mcolor = masks.shape[3]

        # -- torch to numpy --
        device = patches.device
        self._update_gpuid(device)
        naligns = len(blocks[0][0])

        # -- setup batches --
        ps = self.patchsize
        batchsize = self.block_batchsize
        biter = BatchIter(naligns,batchsize)
        block_patches = np.zeros((nimages,nsegs,nframes,batchsize,pcolor,ps,ps))
        block_masks = np.zeros((nimages,nsegs,nframes,batchsize,mcolor,ps,ps))
        block_patches = torch.FloatTensor(block_patches).to(device,non_blocking=True)
        block_masks = torch.FloatTensor(block_masks).to(device,non_blocking=True)
        tokeep = torch.IntTensor(np.arange(naligns)).to(device,non_blocking=True)
        nomotion = torch.LongTensor([4,]*nframes).reshape(1,nframes).to(device)
        nomo = True

        # ---------------------------------------------------
        #   Evaluate Expensive Bootstrapping: One Time Only
        # ---------------------------------------------------

        self.score_cfg.bs_type = "full"
        batch = blocks[:,:,[0],:].to(device)
        scores,scores_t = self.exec_batch(batch,block_patches,patches,
                                          tokeep,nblocks)
        block_utils.block_batch_update(self.samples,scores,
                                       scores_t,batch,K,store_t)
        self.score_cfg.bs_type = "full"

        for batch_indices in biter:

            # -- index tiled images --
            batch = blocks[:,:,batch_indices,:].to(device)
            block_patches_i = block_utils.index_block_batches(block_patches,patches,
                                                              batch,tokeep,
                                                              self.patchsize,
                                                              nblocks,self.gpuid)

            # -- compute directly for sanity check --
            block_patches_i = block_patches_i.to(f'cuda:{self.gpuid}',non_blocking=True)
            # block_masks = block_masks.to(f'cuda:{self.gpuid}',non_blocking=True)
            scores,scores_t = self.compute_batch_scores(block_patches_i,block_masks)
            block_utils.block_batch_update(self.samples,scores,
                                           scores_t,batch,K,store_t)
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

    def score_burst_from_flow(self,burst,flow,patchsize,nblocks):
        if flow.ndim == 4:
            blocks = flow_to_blocks(flow,nblocks)
            blocks = rearrange(blocks,'i p t -> i p 1 t')
        elif flow.ndim == 5:
            nimages,npix,naligns,nframes,two = flow.shape
            flow = rearrange(flow,'i p a t two -> (i a) p t two')
            blocks = flow_to_blocks(flow,nblocks)
            blocks = rearrange(blocks,'(i a) p t -> i p a t',a=naligns)
        else:
            msg = f"Uknown ndims for flow input [{flow.ndim}]. We accept 4 or 5."
            raise ValueError(msg)
        return self.score_burst_from_blocks(burst,blocks,patchsize,nblocks)

    def score_burst_from_blocks(self,burst,blocks,patchsize,nblocks):

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

        masks = np.zeros((nimages,npix,nframes,1,1,patchsize,patchsize))
        scores,scores_t,blocks = self.compute_scores(patches,masks,blocks,
                                                     nblocks,store_t=True)

        # scores.shape = (nimages,nsegs,naligns)
        # scores_t.shape = (nimages,nsegs,naligns,nframes)
        # blocks.shape = (nimages,nsegs,naligns,nframes)

        return scores,scores_t,blocks



