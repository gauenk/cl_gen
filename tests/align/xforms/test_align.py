

# -- python imports 
import torch
import numpy as np
import torchvision.transforms.functional as tvF

# -- project imports --
from align.xforms import align_from_blocks,align_from_flow,align_from_pix

# -- testing imports --
from tests.align.xforms._align_examples import get_example

#
# Testing
# 

    
def test_align():

    def verify_aligned(aligned,burst,ps):
        nframes = aligned.shape[0]
        c_aligned = tvF.center_crop(aligned,(ps,ps))
        ref = tvF.center_crop(burst[nframes//2],(ps,ps))
        for t in range(nframes):
            frame = c_aligned[t,0]
            delta = torch.sum(torch.abs(frame - ref)).item()
            message = f"No error frame [{t}] with values[{c_aligned[t]}] v.s. [{ref}]"
            assert np.isclose(delta,0),message

    def run_check(ex):

        blocks_aligned = align_from_blocks(ex.burst,ex.blocks,
                                           ex.nblocks,
                                           ex.patchsize,
                                           isize=ex.isize)
        verify_aligned(blocks_aligned,ex.burst,ex.patchsize)

        flow_aligned = align_from_flow(ex.burst,ex.flow,
                                       ex.patchsize,
                                       isize=ex.isize)
        verify_aligned(flow_aligned,ex.burst,ex.patchsize)

        pix_aligned = align_from_pix(ex.burst,ex.pix,ex.nblocks)
        verify_aligned(pix_aligned,ex.burst,ex.patchsize)

    def test1():
        ex = get_example("example_1")
        run_check(ex)

    def test2():
        ex = get_example("example_2")
        run_check(ex)

    def test3():
        ex = get_example("example_3")
        run_check(ex)

    def test4():
        ex = get_example("example_4")
        run_check(ex)

    test1()
    test2()
    # test3()
    # test4()


    
