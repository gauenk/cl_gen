
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

    def _update_gpuid(self,device):
        gpuid = device.index
        if gpuid != self.gpuid:
            print(f"Updating EvalScores gpuid from {self.gpuid} to {gpuid}")
            self.gpuid = gpuid
        
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
        if not(scores_t is None):
            scores_t = rearrange(scores_t,'1 (b p) a t -> b p a t',b=nimages)
        scores = rearrange(scores,'1 (b p) a -> b p a',b=nimages)
        return scores,scores_t

    def compute_scores(self,patches,masks,blocks,nblocks,score_t=False):
        # K = len(patches)
        return self.compute_topK_scores(patches,masks,blocks,nblocks,-1,score_t)

    def compute_topK_scores(self,patches,masks,blocks,nblocks,K,score_t=False):
        if isinstance(blocks,BatchGen):
            return self.compute_topK_scores_gen(patches,masks,blocks,nblocks,K,score_t)
        else:
            return self.compute_topK_scores_tensor(patches,masks,blocks,nblocks,K,score_t)

    def exec_batch(self,batch,block_patches,patches,tokeep,nblocks,K):
        # -- index search space  --
        batch = block_samples[:,:,batch_indices,:]                
        batch = batch.to(device)
        # batch.shape = (????)
        #batch = blocks[:,:,batch_indices,:]
        # -- generator so the batch is the batch_index (2nd poor name) --
        # batch = batch_indices.to(device,non_blocking=False)
        batchsize = batch.shape[2]

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
        # -- compute directly for sanity check --
        block_patches_i = block_patches_i.to(f'cuda:{self.gpuid}',
                                             non_blocking=True)
        scores,scores_t = self.compute_batch_scores(block_patches_i,None)
        block_utils.block_batch_update(self.samples,scores,
                                       scores_t,batch,K,score_t)
        torch.cuda.empty_cache()


    @nvtx.annotate("compute_topK_scores", color="orange")
    def compute_topK_scores_gen(self,patches,masks,block_gens,nblocks,K,score_t=False):
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
        self._update_gpuid(device)

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
                batch = batch.to(device)
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
                scores,scores_t = self.compute_batch_scores(block_patches_i,None)
                # scores = scores.cpu()
                # batch = batch.cpu()
                block_utils.block_batch_update(self.samples,scores,
                                               scores_t,batch,K,score_t)
                torch.cuda.empty_cache()

        scores,scores_t,blocks = block_utils.get_topK_samples(self.samples,K)

        # -- no gpu --
        scores = scores.cpu() # scores.shape = (nimages,nsegs,K)
        if torch.is_tensor(scores_t): scores_t = scores_t.cpu() # (nimages,nsegs,K,t)
        blocks = blocks.cpu() # blocks.shape = (nimages,nsegs,K,t)

        return scores,scores_t,blocks
        

    @nvtx.annotate("compute_topK_scores", color="orange")
    def compute_topK_scores_tensor(self,patches,masks,blocks,nblocks,K,score_t=False):
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
        self._update_gpuid(device)

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
            batch = blocks[:,:,batch_indices,:]
            batch = batch.to(device)

            # -- index tiled images --
            block_patches_i = block_utils.index_block_batches(block_patches,patches,
                                                              batch,tokeep,
                                                              self.patchsize,
                                                              nblocks,self.gpuid)

            # -- compute directly for sanity check --
            block_patches_i = block_patches_i.to(f'cuda:{self.gpuid}',non_blocking=True)
            # block_masks = block_masks.to(f'cuda:{self.gpuid}',non_blocking=True)
            scores,scores_t = self.compute_batch_scores(block_patches_i,block_masks)
            block_utils.block_batch_update(self.samples,scores,
                                           scores_t,batch,K,score_t)
        # if nomo is False: print("no motion detected.")
        scores,scores_t,blocks = block_utils.get_topK_samples(self.samples,K)

        # -- no gpu --
        scores = scores.cpu() # scores.shape = (nimages,nsegs,K)
        if torch.is_tensor(scores_t): scores_t = scores_t.cpu() # (nimages,nsegs,K,t)
        blocks = blocks.cpu() # blocks.shape = (nimages,nsegs,K,t)

        return scores,scores_t,blocks

    @nvtx.annotate("compute_topK_scores", color="orange")
    def compute_topK_scores_tensor(self,patches,masks,blocks,nblocks,K,score_t=False):
        self._clear()

        # -- current --
        nimages,nsegs,nframes = patches.shape[:3]
        pcolor,psH,psW = patches.shape[3:]
        mcolor = masks.shape[3]

        # -- torch to numpy --
        device = patches.device
        self._update_gpuid(device)

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

        # -- evaluate with expensive bootstrapping --
        batch = blocks[:,:,[0],:]
        # -- index tiled images --
        block_patches_i = block_utils.index_block_batches(block_patches,patches,
                                                          batch,tokeep,
                                                          self.patchsize,
                                                          nblocks,self.gpuid)
        # -- compute directly for sanity check --
        block_patches_i = block_patches_i.to(f'cuda:{self.gpuid}',non_blocking=True)
        # block_masks = block_masks.to(f'cuda:{self.gpuid}',non_blocking=True)
        scores,scores_t = self.compute_batch_scores(block_patches_i,block_masks)


        for batch_indices in biter:
        #for batch_indices in tqdm.tqdm(biter):

            # -- index search space  --
            batch = blocks[:,:,batch_indices,:]
            batch = batch.to(device)

            # -- index tiled images --
            block_patches_i = block_utils.index_block_batches(block_patches,patches,
                                                              batch,tokeep,
                                                              self.patchsize,
                                                              nblocks,self.gpuid)

            # -- compute directly for sanity check --
            block_patches_i = block_patches_i.to(f'cuda:{self.gpuid}',non_blocking=True)
            # block_masks = block_masks.to(f'cuda:{self.gpuid}',non_blocking=True)
            scores,scores_t = self.compute_batch_scores(block_patches_i,block_masks)
            block_utils.block_batch_update(self.samples,scores,
                                           scores_t,batch,K,score_t)
        # if nomo is False: print("no motion detected.")
        scores,scores_t,blocks = block_utils.get_topK_samples(self.samples,K)

        # -- no gpu --
        scores = scores.cpu() # scores.shape = (nimages,nsegs,K)
        if torch.is_tensor(scores_t): scores_t = scores_t.cpu() # (nimages,nsegs,K,t)
        blocks = blocks.cpu() # blocks.shape = (nimages,nsegs,K,t)

        return scores,scores_t,blocks
        
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
                                                     nblocks,score_t=True)

        # scores.shape = (nimages,nsegs,naligns)
        # scores_t.shape = (nimages,nsegs,naligns,nframes)
        # blocks.shape = (nimages,nsegs,naligns,nframes)

        return scores,scores_t,blocks

    # def score_burst_from_blocks_deprecated(self,burst,blocks,patchsize,nblocks):

    #     # -- assert shape --
    #     assert blocks.ndim == 4,"Must have 4 dims: (nimages,npix,naligns,nframes)"
    #     nimages,npix,naligns,nframes = blocks.shape

    #     # -- get patches --
    #     pad = 2*(nblocks//2)
    #     patches = burst_to_patches(burst,patchsize+pad)

    #     # -- shapes --
    #     device = patches.device
    #     nimages,nsegs,nframes,pcolor,ps,ps = patches.shape
    #     mcolor,batchsize = 1,1

    #     # -- check patchsize and send warning --
    #     if self.patchsize != patchsize:
    #         print("patchsize for execution is not the same as object patchsize")
    #         print(f"Called with {patchsize} but created with {self.patchsize}")

    #     # -- allocate memory --
    #     block_patches = np.zeros((nimages,nsegs,nframes,batchsize,pcolor,ps,ps))
    #     block_masks = np.zeros((nimages,nsegs,nframes,batchsize,mcolor,ps,ps))
    #     block_patches = torch.FloatTensor(block_patches).to(device,non_blocking=True)
    #     block_masks = torch.FloatTensor(block_masks).to(device,non_blocking=True)
    #     tokeep = torch.IntTensor(np.arange(naligns)).to(device,non_blocking=True)
    #     blocks = blocks.to(patches.device)

    #     # -- run blocks --
    #     self._update_gpuid(patches.device)
    #     block_patches_filled = block_utils.index_block_batches(block_patches,patches,
    #                                                       blocks,tokeep,
    #                                                       self.patchsize,
    #                                                       nblocks,self.gpuid)
    #     block_patches_filed = block_patches.to(f'cuda:{self.gpuid}',non_blocking=True)
    #     scores,scores_t = self.compute_batch_scores(block_patches_filled,block_masks)

    #     # -- pick first nalign since its 1 --
    #     scores,scores_t = scores[:,:,0],scores_t[:,:,0,:]

    #     # (nimages,npix) = scores.shape
    #     # (nimages,npix,nframes) = scores_t.shape
    #     return scores,scores_t

        
    
