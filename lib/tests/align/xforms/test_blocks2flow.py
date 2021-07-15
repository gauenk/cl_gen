
# -- python imports 
import torch
import numpy as np

# -- project imports --
from align.xforms import blocks_to_flow

# -- testing imports --
from tests.align.xforms._examples import get_example

#
# Testing
# 

    
def test_block_to_flow():

    def run_check(flow,blocks,nblocks):
        est_flow = blocks_to_flow(blocks,nblocks)
        delta = torch.sum(torch.abs(est_flow-flow)).item()
        assert np.isclose(delta,0),f"No error [{est_flow}] v.s. [{flow}]"

    def test1():
        ex = get_example("example_1")
        run_check(ex.flow,ex.blocks,ex.nblocks)

    def test2():
        ex = get_example("example_2")
        run_check(ex.flow,ex.blocks,ex.nblocks)

    def test3():
        ex = get_example("example_3")
        run_check(ex.flow,ex.blocks,ex.nblocks)

    def test4():
        ex = get_example("example_4")
        run_check(ex.flow,ex.blocks,ex.nblocks)

    test1()
    test2()
    test3()
    test4()

