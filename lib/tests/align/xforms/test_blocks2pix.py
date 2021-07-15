
# -- imports --

# -- python imports --
import torch
import numpy as np

# -- project imports --
from align.xforms import blocks_to_pix

# -- testing imports --
from tests.align.xforms._examples import get_example

#
# Testing
# 

def test_blocks_to_pix():
    """
    pix is the [x,y] coordinate of the nnf in a frame t

    pix.shape = (num of images, num of pixels, num of frames, 2)

    pix[i,t] is the [x,y] coordinate mapping to 
    the pix


    """
    def run_check(pix,blocks,nblocks,centers):
        est_pix = blocks_to_pix(blocks,nblocks,centers=centers)
        delta = torch.sum(torch.abs(est_pix-pix)).item()
        assert np.isclose(delta,0),f"Neq pixels: [{est_pix}] v.s. [{pix}]"

    def test1():
        ex = get_example("example_1")
        run_check(ex.pix,ex.blocks,ex.nblocks,ex.centers)

    def test2():
        ex = get_example("example_2")
        run_check(ex.pix,ex.blocks,ex.nblocks,ex.centers)

    def test3():
        ex = get_example("example_3")
        run_check(ex.pix,ex.blocks,ex.nblocks,ex.centers)

    def test4():
        ex = get_example("example_4")
        run_check(ex.pix,ex.blocks,ex.nblocks,ex.centers)

    test1()
    test2()
    test3()
    test4()
