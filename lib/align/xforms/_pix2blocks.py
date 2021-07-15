
# -- python imports --
import numpy as np
from einops import rearrange

# -- numba imports --
from numba import jit,prange

# -- pytorch imports --
import torch

# -- project imports --
from align.xforms._pix2flow import pix_to_flow
from align.xforms._flow2blocks import flow_to_blocks

def pix_to_blocks(pix,nblocks):
    flow = pix_to_flow(pix)
    blocks = flow_to_blocks(flow,nblocks)
    return blocks
