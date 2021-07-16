
# -- python imports --
import torch
import numpy as np

# -- project imports --
from align.xforms import pix_to_blocks

# -- testing imports --
from tests.align.xforms._onepix_examples import get_example

#
# Testing
# 

def test_pix_to_blocks():
    """
    pix is the [x,y] coordinate of the nnf in a frame t

    pix.shape = (num of images, num of pixels, num of frames, 2)

    pix[i,t] is the [x,y] coordinate mapping to 
    the pix


    """
    def run_check(pix,blocks,nblocks):
        est_blocks = pix_to_blocks(pix,nblocks)
        delta = torch.sum(torch.abs(est_blocks-blocks)).item()
        assert np.isclose(delta,0),f"No error {est_blocks} vs {blocks}"

    def test1():
        ex = get_example("example_1")
        run_check(ex.pix,ex.blocks,ex.nblocks)

    def test2():
        ex = get_example("example_2")
        run_check(ex.pix,ex.blocks,ex.nblocks)

    def test3():
        ex = get_example("example_3")
        run_check(ex.pix,ex.blocks,ex.nblocks)

    def test4():
        ex = get_example("example_4")
        run_check(ex.pix,ex.blocks,ex.nblocks)

    test1()
    test2()
    test3()
    test4()
