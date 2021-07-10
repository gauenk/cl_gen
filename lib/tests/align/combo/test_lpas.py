
from easydict import EasyDict as edict

from pyutils import create_named_meshgrid
#from .create_patches import patches_v1

import align


def create_test_grid():
    pass

# def test_pixel_alignment():

#     # -- dataset grid --
#     Tgrid,Bgrid = [2,3,10],[1,2,10]
#     Cgrid,Hgrid,Wgrid = [3],[64],[64]
#     eps_grid,std_grid = [0.,25.,50.],[0.,50.,100.]
#     lists = [Tgrid,Bgrid,Cgrid,Hgrid,Wgrid,eps_grid,std_grid]
#     order = ['T','B','C','H','W','eps','std']
#     ds_configs = create_named_meshgrid(lists,order)

#     # -- parameter grid --
#     nblocks,patchsize = [3,9],[3,9,32]
#     rand_seed = [123,234]
#     motion_type = ["global"]
#     lists = [patchsize,nblocks]
#     order = ['patchsize','nblocks','rand_seed','motion_type']
#     param_configs = create_named_meshgrid(lists,order)

#     align_cfg = edict()
#     align_str = "c-lpas"

#     for ds_cfg in ds_configs:
#         for ds_cfg in ds_configs:
#             inputs = [test[k] for k in order]
#             images = patches_v1(*inputs)
#             results = align.compute_alignment(cfg,align_str,images)
        
