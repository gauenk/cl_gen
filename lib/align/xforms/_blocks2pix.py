
# -- python imports --
import numpy as np
from einops import rearrange,repeat

# -- numba imports --
from numba import jit,prange

# -- pytorch imports --
import torch

# -- project imports --
# from align.xforms import flow_to_pix,blocks_to_flow
from ._flow2pix import flow_to_pix
from ._blocks2flow import blocks_to_flow

def blocks_to_pix(blocks,nblocks,centers=None):
    flow = blocks_to_flow(blocks,nblocks)
    pix = flow_to_pix(flow,centers)
    return pix

