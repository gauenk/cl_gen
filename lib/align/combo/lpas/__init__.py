"""
Local patch alignment search

"""

from ._utils import assert_cfg_fields
from .spoof import spoof

def search(cfg,burst):
    pass



def choose_mtype(mtype):
    if mtype == "global":
        # patchsize doesn't matter
        pass
    elif mtype == "masked":
        # given a mask per image
        pass
    elif mtype == "pixel":
        # each pixel is computed separately
        pass
