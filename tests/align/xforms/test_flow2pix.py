
# -- python imports --
import torch
import numpy as np

# -- project imports --
from align.xforms import flow_to_pix,compute_pix_delta

# -- testing imports --
from tests.align.xforms._onepix_examples import get_example as get_onepix_example
from tests.align.xforms._align_examples import get_example as get_multipix_example
#
# Testing
# 

def test_flow_to_pix_centers():
    """
    pix is the [x,y] coordinate of the nnf in a frame t

    pix.shape = (num of images, num of pixels, num of frames, 2)

    pix[i,t] is the [x,y] coordinate mapping to 
    the pix


    """
    def run_check(pix,flow,centers,isize):
        est_pix = flow_to_pix(flow,centers=centers)
        delta = compute_pix_delta(est_pix,pix)
        # delta = torch.sum(torch.abs(est_pix-pix)).item()
        msg = f"[Using centers] No error {torch.cat([est_pix,pix],dim=-1)}"
        assert np.isclose(delta,0),msg

    def test1():
        ex = get_onepix_example("example_1")
        run_check(ex.pix,ex.flow,ex.centers,ex.isize)

        ex = get_multipix_example("example_1")
        run_check(ex.pix,ex.flow,ex.centers,ex.isize)

    def test2():
        ex = get_onepix_example("example_2")
        run_check(ex.pix,ex.flow,ex.centers,ex.isize)

    def test3():
        ex = get_onepix_example("example_3")
        run_check(ex.pix,ex.flow,ex.centers,ex.isize)

    def test4():
        ex = get_onepix_example("example_4")
        run_check(ex.pix,ex.flow,ex.centers,ex.isize)


    test1()
    test2()
    test3()
    test4()

def test_flow_to_pix_isize():
    """
    pix is the [x,y] coordinate of the nnf in a frame t

    pix.shape = (num of images, num of pixels, num of frames, 2)

    pix[i,t] is the [x,y] coordinate mapping to 
    the pix


    """
    def run_check(pix,flow,centers,isize):
        est_pix = flow_to_pix(flow,isize=isize)
        delta = compute_pix_delta(est_pix,pix)
        # delta = torch.sum(torch.abs(est_pix-pix)).item()
        msg = f"[Using isize] No error {torch.stack([est_pix,pix])}"
        assert np.isclose(delta,0),msg

    def test1():
        ex = get_onepix_example("example_1")
        pix = ex.pix - ex.centers
        run_check(pix,ex.flow,ex.centers,ex.isize)

    def test2():
        ex = get_onepix_example("example_2")
        pix = ex.pix - ex.centers
        run_check(pix,ex.flow,ex.centers,ex.isize)

    def test3():
        ex = get_onepix_example("example_3")
        pix = ex.pix - ex.centers
        run_check(pix,ex.flow,ex.centers,ex.isize)

    def test4():
        ex = get_onepix_example("example_4")
        pix = ex.pix - ex.centers
        run_check(pix,ex.flow,ex.centers,ex.isize)


    test1()
    test2()
    test3()
    test4()

