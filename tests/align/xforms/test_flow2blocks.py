# -- python imports --
import torch
import numpy as np

# -- project imports --
from align.xforms import flow_to_blocks

# -- python imports 
import torch
import numpy as np

# -- project imports --
from align.xforms import flow_to_blocks

# -- testing imports --
from tests.align.xforms._onepix_examples import get_example



def test_flow_to_block():
    """
    flow is the [dx,dy] motion of the _object_ in a frame

    """
    def run_check(flow,blocks,nblocks):
        est_blocks = flow_to_blocks(flow,nblocks)
        print(est_blocks)
        delta = torch.sum(torch.abs(est_blocks-blocks)).item()
        assert np.isclose(delta,0),f"No error {est_blocks} vs {blocks}"

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

    def test5():
        ex = get_example("example_5")
        run_check(ex.flow,ex.blocks,ex.nblocks)

    test1()
    test2()
    test3()
    test4()
    test5()
