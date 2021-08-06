
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
from align._utils import BatchIter
from align.xforms import blocks_to_pix

class EvalBlockScores():

    def __init__(self,score_fxn,score_fxn_name,patchsize,
                 block_batchsize,noise_info,gpuid=1):
        self.score_fxn = score_fxn
        self.score_fxn_name = score_fxn_name
        self.patchsize = patchsize
        self.block_batchsize = block_batchsize
        self.noise_info = noise_info
        self.indexing = edict({})
        self.samples = edict({'scores':[],'blocks':[]})
        self.gpuid = gpuid

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
        cfg = edict({'gpuid':self.gpuid})
        scores,scores_t = self.score_fxn(cfg,patches)
        scores = rearrange(scores,'1 (b p) a -> b p a',b=nimages)
        return scores

    def compute_scores(self,patches,masks,blocks):
        K = len(patches)
        return self.compute_topK_scores(patches,masks,blocks,K)

    def compute_topK_scores(self,patches,masks,blocks,nblocks,K):
        if isinstance(blocks,BatchGen):
            return self.compute_topK_scores_gen(patches,masks,blocks,nblocks,K)
        else:
            return self.compute_topK_scores_tensor(patches,masks,blocks,nblocks,K)

    @nvtx.annotate("compute_topK_scores", color="orange")
    def compute_topK_scores_gen(self,patches,masks,block_gens,nblocks,K):
        r"""
        Generator
        """
        self._clear()
        # -- current --
        nimages,nsegs,nframes = patches.shape[:3]
        pcolor,psH,psW = patches.shape[3:]
        mcolor = masks.shape[3]

        # -- torch to numpy --
        device = patches.device

        # -- setup batches --
        ps = self.patchsize
        batchsize = self.block_batchsize
        naligns = batchsize
        #biter = BatchIter(naligns,batchsize)
        block_patches = np.zeros((nimages,nsegs,nframes,batchsize,pcolor,ps,ps))
        block_masks = np.zeros((nimages,nsegs,nframes,batchsize,mcolor,ps,ps))
        # print("patches.device", patches.device)
        # print(torch.cuda.memory_summary(0))
        # print(torch.cuda.memory_summary(1))
        # print(torch.cuda.memory_summary(2))
        block_patches = torch.FloatTensor(block_patches).to(device,non_blocking=True)
        block_masks = torch.FloatTensor(block_masks).to(device,non_blocking=True)
        tokeep = torch.IntTensor(np.arange(naligns)).to(device,non_blocking=True)
        nomotion = torch.LongTensor([4,]*nframes).reshape(1,nframes).to(device)
        nomo = True

        # total = 25**4
        # bs = 128
        # nbatches = total / bs.
        idx = -1
        nbgens = len(block_gens)
        # print("nbgens.",nbgens)
        #for block_gen_samples in tqdm.tqdm(block_gens,total=nbgens):
        for block_gen_samples in block_gens:
            idx += 1
            naligns = block_gen_samples.shape[2]
            # print("block_gen_samples.shape ",block_gen_samples.shape)
            biter = BatchIter(naligns,self.block_batchsize)
            for batch_indices in biter:
                # print(f"batch index: {idx}")
                
                # -- index search space  --
                batch = block_gen_samples[:,:,batch_indices,:]
                # batch.shape = (????)
                #batch = blocks[:,:,batch_indices,:]
                # -- generator so the batch is the batch_index (2nd poor name) --
                # batch = batch_indices.to(device,non_blocking=False)
                batchsize = batch.shape[2]
                
                # -- find no motion --
                check = torch.sum(torch.abs(batch[0][0] - nomotion),dim=1)
                args = torch.where(check == 0)
                if len(args[0]) > 0: nomo = False
                
                # -- index tiled images --
                # -- [OLD CODE] --
                # block_patches_i = block_patches[:,:,:,:batchsize]
                # block_masks_i = block_masks[:,:,:,:batchsize]
                # bp_shape = block_patches.shape
                # naligns = bp_shape[3]

                # print(f"[a] block_patches.shape {bp_shape} | naligns {naligns}")

                # -- [OLD CODE] --
                # block_patches_nba = numba.cuda.as_cuda_array(block_patches_i)
                # patches_nba = numba.cuda.as_cuda_array(patches)
                # batch_nba = numba.cuda.as_cuda_array(batch)
                # block_utils.index_block_batches(block_patches_nba,patches_nba,batch_nba,
                #                                 self.patchsize,nblocks)
                
                block_patches_i = block_utils.index_block_batches(block_patches,patches,
                                                                  batch,tokeep,
                                                                  self.patchsize,
                                                                  nblocks,self.gpuid)

                
                # print(batch.shape)
                # print(block_patches_i.shape)
                # print("-=-=-"*3)
                # print("pre.")
                # print("-=-=-"*3)
                # print("CUDA:0")
                # print(torch.cuda.list_gpu_processes('cuda:0'))
                # print("CUDA:1")
                # print(torch.cuda.list_gpu_processes('cuda:1'))

                # -- compute directly for sanity check --
                block_patches_i = block_patches_i.to(f'cuda:{self.gpuid}',
                                                     non_blocking=True)
                # block_masks_i = block_masks_i.to('cuda:1',non_blocking=True)

                # print("-=-=-"*3)
                # print("mid. [a]")
                # print("-=-=-"*3)
                # print("CUDA:0")
                # print(torch.cuda.list_gpu_processes('cuda:0'))
                # print("CUDA:1")
                # print(torch.cuda.list_gpu_processes('cuda:1'))

                scores = self.compute_batch_scores(block_patches_i,None)#block_masks_i)
                
                # print("-=-=-"*3)
                # print("mid.")
                # print("-=-=-"*3)
                # print("CUDA:0")
                # print(torch.cuda.list_gpu_processes('cuda:0'))
                # print("CUDA:1")
                # print(torch.cuda.list_gpu_processes('cuda:1'))

                scores = scores.cpu()
                batch = batch.cpu()
                block_utils.block_batch_update(self.samples,scores,batch,K)

                # print("-=-=-"*3)
                # print("post.")
                # print("-=-=-"*3)
                # print("CUDA:0")
                # print(torch.cuda.list_gpu_processes('cuda:0'))
                # print("CUDA:1")
                # print(torch.cuda.list_gpu_processes('cuda:1'))

                torch.cuda.empty_cache()

        # if nomo is False: print("[gens] no motion detected.")
        # print("done.")
        scores,blocks = block_utils.get_block_batch_topK(self.samples,K)
        return scores,blocks
        

    @nvtx.annotate("compute_topK_scores", color="orange")
    def compute_topK_scores_tensor(self,patches,masks,blocks,nblocks,K):
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

        #[old] blocks.shape = (nimage,nsegs,nframes,narrangements)
        blocks.shape = (nimage,nsegs,narrangements, nframes)
        """

        self._clear()
        # -- current --
        nimages,nsegs,nframes = patches.shape[:3]
        pcolor,psH,psW = patches.shape[3:]
        mcolor = masks.shape[3]

        # -- torch to numpy --
        device = patches.device
        naligns = len(blocks[0][0])
        # naligns = blocks.shape[2] # NOT a tensor!

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

        for batch_indices in biter:
        #for batch_indices in tqdm.tqdm(biter):

            # -- index search space  --
            # batch = index_blocks_with_batches(blocks,batch_indices)
            batch = blocks[:,:,batch_indices,:]

            # -- find no motion --
            # check = torch.sum(torch.abs(batch[0][0] - nomotion),dim=1)
            # args = torch.where(check == 0)
            # if len(args[0]) > 0: nomo = False

            # -- index tiled images --
            block_patches_i = block_utils.index_block_batches(block_patches,patches,
                                                              batch,tokeep,
                                                              self.patchsize,
                                                              nblocks,self.gpuid)

            # -- compute directly for sanity check --
            block_patches_i = block_patches_i.to(f'cuda:{self.gpuid}',non_blocking=True)
            # block_masks = block_masks.to('cuda:1',non_blocking=True)
            scores = self.compute_batch_scores(block_patches_i,block_masks)

            block_utils.block_batch_update(self.samples,scores,batch,K)
        # if nomo is False: print("no motion detected.")
        scores,blocks = block_utils.get_block_batch_topK(self.samples,K)
        return scores,blocks
        
        
