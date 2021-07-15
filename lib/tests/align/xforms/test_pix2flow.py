
# -- python imports --
import torch
import numpy as np

# -- project imports --
from align.xforms import pix_to_flow

# -- testing imports --
from tests.align.xforms._examples import get_example

#
# Testing
# 

def test_pix_to_flow():
    """
    pix is the [x,y] coordinate of the nnf in a frame t

    pix.shape = (num of images, num of pixels, num of frames, 2)

    pix[i,t] is the [x,y] coordinate mapping to 
    the pix


    """
    def run_check(pix,flow):
        est_flow = pix_to_flow(pix)
        delta = torch.sum(torch.abs(est_flow-flow)).item()
        assert np.isclose(delta,0),f"No error {est_flow} vs {flow}"

    def test1():
        ex = get_example("example_1")
        run_check(ex.pix,ex.flow)

    def test2():
        ex = get_example("example_2")
        run_check(ex.pix,ex.flow)

    def test3():
        ex = get_example("example_3")
        run_check(ex.pix,ex.flow)

    def test4():
        ex = get_example("example_4")
        run_check(ex.pix,ex.flow)


    test1()
    test2()
    test3()
    test4()
