
# -- python imports --
from easydict import EasyDict as edict
from einops import rearrange

# -- pytorch imports --
import torch

# -- project imports --
from align._utils import torch_to_numpy
import align.combo._block_utils as block_utils

class EvalBlockScores():

    def __init__(self,score_fxn,patchsize,block_batchsize,noise_info):
        self.score_fxn = score_fxn
        self.patchsize = patchsize
        self.block_batchsize = block_batchsize
        self.noise_info = noise_info
        self.indexing = edict({})
        self.samples = edict({'scores':[],'blocks':[]})

    def _clear(self):
        self.samples = edict({'scores':[],'blocks':[]})
        
    def compute_batch_scores(self,patches,masks):
        patches = rearrange(patches,'b s t a c h w -> s b a t c h w')
        scores,scores_t = self.score_fxn(None,patches)
        scores = rearrange(scores,'s b a -> b s a')
        return scores

    def compute_scores(self,patches,masks,blocks):
        K = len(patches)
        return self.compute_topK_scores(patches,masks,blocks,K)

    def compute_topK_scores(self,patches,masks,blocks,nblocks,K):
        """

        Search Space right now = total # of patches
        
        Goal:
        - given a set of patches and block arangements

        Old:
        - patches tiled into "H" arrangements
        - "blocks" subsets different orderings of "H" from each "T" frames
        - inner-loop evals over the indexed region, patches[:,:,:,blocks,:,:,:]

        Update:
        - The "H" (arrangements) is implicit
        - The "R" (num_nl_patches) can be grouped with "C,pH,pW" into "F": features
        
        Thinking:
        - Vectorized ops are faster
        - Rearrange requires mysterious computation overhead
        - Each patch has different subset of the H^T arrangements

        Restriction:
        - compute_score requires specific image shape

        Question: How do we want to split/vectorize pixels?
        - FAISS uses batches to run kNN; we should do the same
          - Batches of pixels means ref_t uses "batch" and all other frames remain.
          - Simple case is to batch them all.
        
        Todo:
        - add another dimension to "blocks" for each ref_t pixel (later a batch).
        - index patches with "blocks" using implicit H space.

        blocks.shape = (nimage,nsegs,nframes,narrangements)
        """

        self._clear()
        # -- current --
        nimages,nsegs,nframes = patches.shape[:3]
        # nimages,nsegs,naligns,nframes = blocks.shape

        # # -- old --
        # nimages,num_nl_patches,nframes = patches.shape[:3]
        # search_space,C,H,W = patches.shape[3:]
        # B,R,T,H,C,pH,pW = patches.shape

        # -- torch to numpy --
        device = patches.device
        patches = torch_to_numpy(patches)
        masks = torch_to_numpy(masks)
        blocks = torch_to_numpy(blocks)

        for batch in block_utils.iter_block_batches(blocks,self.block_batchsize):

            block_patches = block_utils.index_block_batches(patches,batch,
                                                            self.patchsize,nblocks)
            block_masks = block_utils.index_block_batches(masks,batch,
                                                          self.patchsize,nblocks)
            # -- numpy to torch --
            block_patches = torch.Tensor(block_patches).to(device)

            block_masks = torch.Tensor(block_masks).to(device)
            
            scores = self.compute_batch_scores(block_patches,block_masks)
            block_utils.block_batch_update(self.samples,scores,batch,K)
        scores,blocks = block_utils.get_block_batch_topK(self.samples,K)
        return scores,blocks
        
        
