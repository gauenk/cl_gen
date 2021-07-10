
from align.xforms import global_flow_to_blocks,global_blocks_to_flow
import torch
import numpy as np

#
# Testing
# 

def test_global_flow_to_block():
    """
    flow is the [dx,dy] motion of the _object_ in a frame

    """
    def run_check(flow,blocks,nblocks):
        est_blocks = global_flow_to_blocks(flow,nblocks)
        delta = torch.sum(torch.abs(est_blocks-blocks)).item()
        assert np.isclose(delta,0),f"No error {est_blocks} vs {blocks}"

    def test1():
        flow = torch.LongTensor([[[1,-1],[0,-1]]])
        blocks = torch.LongTensor([[8,4,1]])
        # blocks = torch.LongTensor([[0,4,7]])
        nblocks = 3
        run_check(flow,blocks,nblocks)

    def test2():
        flow = torch.LongTensor([[[1,1],[0,-1]]])
        blocks = torch.LongTensor([[2,4,1]])
        nblocks = 3
        run_check(flow,blocks,nblocks)

    def test3():
        flow = torch.LongTensor([[[1,-1],[0,-1],[1,0]]])
        # blocks = torch.LongTensor([[1,7,12,13]])
        blocks = torch.LongTensor([[23,17,12,11]])
        nblocks = 5
        run_check(flow,blocks,nblocks)

    test1()
    test2()
    test3()
    
def test_global_block_to_flow():
    def run_check(flow,blocks,nblocks):
        est_flow = global_blocks_to_flow(blocks,nblocks)
        delta = torch.sum(torch.abs(est_flow-flow)).item()
        assert np.isclose(delta,0),"No error"

    def test1():
        flow = torch.LongTensor([[[1,-1],[0,-1]]])
        blocks = torch.LongTensor([[8,4,1]])
        nblocks = 3
        run_check(flow,blocks,nblocks)

    def test2():
        flow = torch.LongTensor([[[1,1],[0,-1]]])
        blocks = torch.LongTensor([[2,4,1]])
        nblocks = 3
        run_check(flow,blocks,nblocks)

    def test3():
        flow = torch.LongTensor([[[1,-1],[0,-1],[1,0]]])
        blocks = torch.LongTensor([[23,17,12,11]])
        nblocks = 5
        run_check(flow,blocks,nblocks)

    test1()
    test2()
    test3()

