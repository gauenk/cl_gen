
# -- python imports --
import time,nvtx
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch imports --
import torch

# -- project imports --
from pyutils import torch_to_numpy,save_image,images_to_psnrs
import align.combo._block_utils as block_utils
from align._utils import BatchIter
from align.xforms import blocks_to_pix

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
        # -- [old] each npix is computed together --
        # patches = rearrange(patches,'b s t a c h w -> s b a t c h w')
        # scores,scores_t = self.score_fxn(None,patches)
        # scores = rearrange(scores,'s b a -> b s a').cpu()

        # -- [new] each npix is computed separately --
        nimages = patches.shape[0]
        patches = rearrange(patches,'b p t a c h w -> 1 (b p) a t c h w')
        scores,scores_t = self.score_fxn(None,patches)
        scores = rearrange(scores,'1 (b p) a -> b p a',b=nimages).cpu()
        return scores

    def compute_scores(self,patches,masks,blocks):
        K = len(patches)
        return self.compute_topK_scores(patches,masks,blocks,K)

    @nvtx.annotate("compute_topK_scores", color="orange")
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
        pcolor,psH,psW = patches.shape[3:]
        mcolor = masks.shape[3]
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
        naligns = blocks.shape[2]

        # -- setup batches --
        ps = self.patchsize
        batchsize = self.block_batchsize
        biter = BatchIter(naligns,batchsize)
        block_patches = np.zeros((nimages,nsegs,nframes,batchsize,pcolor,ps,ps))
        block_masks = np.zeros((nimages,nsegs,nframes,batchsize,mcolor,ps,ps))
        

        def sanity(patches):
            print("patches.shape ",patches.shape)
            nimages = patches.shape[0]
            nframes = patches.shape[2]
            patches = rearrange(patches,'b p t a c h w -> 1 (b p) a t (c h w)')
            rshape = 'r bp a 1 chw -> r bp a t chw'
            ref = repeat(patches[:,:,:,[nframes//2]],rshape,t=nframes)
            ave = torch.mean((ref - patches)**2,dim=(-2,-1))
            print("ave.shape ",ave.shape)
            rshape = '1 (b p) a -> b p a'
            scores = rearrange(ave,rshape,b=nimages).cpu()
            return scores

        
        # blocks_rs = rearrange(blocks,'i p t h -> (i h) p t')
        # print("blocks.shape", blocks.shape)
        # print("blocks_rs.shape", blocks_rs.shape)
        # pad = 2*(nblocks//2)
        # isize = edict({'h':ps+pad,'w':ps+pad})
        # pix_init = blocks_to_pix(blocks_rs,nblocks,isize=isize)
        # print("pix_init.shape", pix_init.shape)
        # pix = rearrange(pix_init,'(i h) p t two -> i p t h two')
        # print(pix.shape)

        #for batch in block_utils.iter_block_batches(blocks,self.block_batchsize):

        # -- debug --
        fs = int(np.sqrt(nsegs))
        is_even = (fs%2) == 0
        # mid_pix = fs*fs//2 + (fs//2)*is_even
        mid_pix = 32*10+23
        # print(mid_pix,mid_pix//fs,mid_pix%fs)

        for batch_indices in biter:

            # -- index search space  --
            # batch = pix[:,:,batch_indices]
            batch = blocks[:,:,batch_indices,:]
            batchsize = batch.shape[2]
            # print(batch[0,-1])
            # print(batch[0,mid_pix])

            # -- index tiled images --
            # block_patches = block_utils.index_block_batches(patches,batch,
            #                                                 self.patchsize,nblocks)
            # block_masks = block_utils.index_block_batches(masks,batch,
            #                                               self.patchsize,nblocks)
            block_patches = torch_to_numpy(block_patches)[:,:,:,:batchsize]
            block_masks = torch_to_numpy(block_masks)[:,:,:,:batchsize]
            block_utils.index_block_batches(block_patches,patches,batch,
                                            self.patchsize,nblocks)
            # block_utils.index_block_batches(block_masks,masks,batch,
            #                                 self.patchsize,nblocks)

            # -- numpy to torch --
            block_patches = torch.Tensor(block_patches).to(device)
            # block_masks = torch.Tensor(block_masks).to(device)
            
            # -- compute directly for sanity check --
            scores = self.compute_batch_scores(block_patches,block_masks)
            # scores = sanity(block_patches)
            # print(torch_to_numpy(scores[0,-1]))
            # print(torch_to_numpy(scores[0,mid_pix]))
            # ref = repeat(block_patches[0,-1,nframes//2,12],
            #              'c h w -> a c h w',a=batchsize)
            # for t in range(nframes):
            #     print(f"[{t}] psnrs ",images_to_psnrs(block_patches[0,-1,t],ref))
            # print(torch.topk(scores[0,-1],1,largest=False))
            # print(torch.topk(scores[0,mid_pix],1,largest=False))
            # print(torch_to_numpy(scores[0,-1]))
            # # shape = (b p t a c h w)
            # save_image(torch.FloatTensor(patches[0,-1]),"pactches.png")
            # save_image(block_patches[0,-1,2],"arangements.png")
            # print("sleeping.")
            # time.sleep(3)


            block_utils.block_batch_update(self.samples,scores,batch,K)
        scores,blocks = block_utils.get_block_batch_topK(self.samples,K)
        return scores,blocks
        
        
