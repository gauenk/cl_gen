# -- python imports --
import numpy as np

# -- pytorch imports --
import torch

def global_blocks_ref_to_frames(ref_blocks_i,nblocks):
    r"""

    converts the block positions w.r.t a reference frame
    into block positions w.r.t. each individual frame

    e.g.

    nimages = 1
    nframes = 3
    nblocks = 3
    patchsize = 3

    ref_blocks = [[1,4,2]]

    ref_blocks (ndarray)
       shape (nimages,nframes)
    the block representing the location of the 

    nblocks (int)
       number of blocks along one dimension

    """
    print("DECREPATED [07-13-2021]: This function doesn't make sense to gauenk.")
    ref_blocks = ref_blocks_i.clone()
    nbatches,nframes = ref_blocks.shape[:2]
    ref_t = nframes//2
    grid = np.arange(nblocks**2).reshape(nblocks,nblocks)
    frame_blocks = []
    ref_blocks = ref_blocks.cpu().numpy()
    for b in range(nbatches):
        frame_blocks_b = []
        blocks = ref_blocks[b]
        ref = blocks[ref_t]

        # -- before ref --
        left = blocks[:ref_t]
        dleft = ref - left
        left += 2*dleft

        # -- after ref --
        right = blocks[ref_t+1:]
        dright = right - ref
        right -= 2*dright

        frame_blocks_b = np.r_[left,ref,right]
        frame_blocks.append(frame_blocks_b)
    frame_blocks = torch.LongTensor(frame_blocks)
    return frame_blocks
