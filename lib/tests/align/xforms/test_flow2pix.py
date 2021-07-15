
# -- python imports --
import torch
import numpy as np

# -- project imports --
from align.xforms import flow_to_pix

# -- testing imports --
from tests.align.xforms._examples import get_example

#
# Testing
# 

def test_flow_to_pix():
    """
    pix is the [x,y] coordinate of the nnf in a frame t

    pix.shape = (num of images, num of pixels, num of frames, 2)

    pix[i,t] is the [x,y] coordinate mapping to 
    the pix


    """
    def run_check(pix,flow,centers,isize):
        est_pix = flow_to_pix(flow,centers=centers)
        delta = torch.sum(torch.abs(est_pix-pix)).item()
        assert np.isclose(delta,0),f"[Using centers] No error {est_pix} vs {pix}"

        est_pix = flow_to_pix(flow,isize=isize)
        delta = torch.sum(torch.abs(est_pix-pix)).item()
        assert np.isclose(delta,0),f"[Not using centers] No error {est_pix} vs {pix}"

    def test1():
        ex = get_example("example_1")
        run_check(ex.pix,ex.flow,ex.centers,ex.isize)

    def test2():
        ex = get_example("example_2")
        run_check(ex.pix,ex.flow,ex.centers,ex.isize)

    def test3():
        ex = get_example("example_3")
        run_check(ex.pix,ex.flow,ex.centers,ex.isize)

    def test4():
        ex = get_example("example_4")
        run_check(ex.pix,ex.flow,ex.centers,ex.isize)


    test1()
    test2()
    test3()
    test4()
