
def compute_topK_scores_gen(self,patches,masks,biter,nblocks,K):
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
    block_patches = torch.FloatTensor(block_patches).to(device,non_blocking=True)
    block_masks = torch.FloatTensor(block_masks).to(device,non_blocking=True)

    for idx,batch_indices in enumerate(biter):
        print(f"batch index: {idx}")

        # -- index search space  --
        # batch = blocks[:,:,batch_indices,:]
        # -- generator so the batch is the batch_index (2nd poor name) --
        batch = batch_indices.to(device,non_blocking=True)
        batchsize = batch.shape[2]
        
        # -- indexing shared, read-only memory --
        block_patches_i = block_patches[:,:,:,:batchsize]
        block_masks_i = block_masks[:,:,:,:batchsize]
        bp_shape = block_patches.shape
        naligns = bp_shape[3]

        # -- further indexing shared, read-only memory --
        block_patches_nba = numba.cuda.as_cuda_array(block_patches_i)
        patches_nba = numba.cuda.as_cuda_array(patches)
        batch_nba = numba.cuda.as_cuda_array(batch)
        block_utils.index_block_batches(block_patches_nba,patches_nba,batch_nba,
                                        self.patchsize,nblocks)
        
        # -- move data to gpu for faster compute. read-only memory --
        block_patches_i = block_patches_i.to('cuda:1',non_blocking=True)
        block_masks_i = block_masks_i.to('cuda:1',non_blocking=True)
        scores = self.compute_batch_scores(block_patches_i,block_masks_i)

        # -- move data to gpu for faster compute. --
        # -- write data to shared memory; only requires K writes --
        block_utils.block_batch_update(self.samples,scores,batch,K)

    print("done.")
    scores,blocks = block_utils.get_block_batch_topK(self.samples,K)
    return scores,blocks


