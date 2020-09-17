"""
We wrap the entire model into the denoising block 
to allow for DistributedDataParallel
"""

import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F

# local imports
from .utils import share_encoding_mean


class DenoisingBlock(nn.Module):

    def __init__(self,encoder,decoder,projector,device,
                 num_transforms,batch_size,agg_fxn,agg_type):
        super(DenoisingBlock, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector
        
        self.device = device
        self.num_transforms = num_transforms
        self.batch_size = batch_size
        
        self._agg_fxn = agg_fxn
        self._agg_type = agg_type

    def forward(self,pic_set):
        simh = []
        simpics = []

        N = len(pic_set)
        BS = pic_set[0].shape[0]
        pshape = pic_set[0][0].shape
        shape = (N,BS,) + pshape

        # encode
        pic_set = pic_set.reshape((N*BS,)+pshape)
        h,skip = self.encoder(pic_set)
        proj = self.projector(h).reshape(N,BS,-1)
        
        # aggregate
        agg_h,agg_skip = self.aggregate(h,skip,N,BS)

        # decode
        input_i = [agg_h,agg_skip]
        dec_pics = self.decoder(input_i)
        dec_pics = dec_pics.reshape((N,BS,) + pshape)

        return dec_pics,proj

    def aggregate(self,h,skip,N,BS):
        agg_fxn = self._agg_fxn
        agg_type = self._agg_type
        if agg_fxn == 'mean':
            return share_encoding_mean(agg_type,h,skip,N,BS)
        elif agg_fxn == 'id':
            return h,skip
        else:
            raise ValueError(f"Uknown aggregation function [{agg_fxn}]")
