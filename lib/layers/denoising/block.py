"""
We wrap the entire model into the denoising block 
to allow for DistributedDataParallel
"""

import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import ResNet

# local imports
from .utils import share_encoding_mean
from layers.resnet import ResNetWithSkip



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

        if isinstance(self.encoder,ResNetWithSkip):
            self.two_outputs = True
            self.offset_pic_set = True
        elif isinstance(self.encoder,ResNet):
            self.two_outputs = False
            self.offset_pic_set = True
        else:
            self.two_outputs = True
            self.offset_pic_set = False

    def forward(self,pic_set):
        simh = []
        simpics = []

        N = len(pic_set)
        BS = pic_set[0].shape[0]
        pshape = pic_set[0][0].shape
        shape = (N,BS,) + pshape


        if self.offset_pic_set:
            enc_inputs = pic_set
            # pic_set = pic_set.view((N*BS,-1))
            # enc_inputs = pic_set - torch.min(pic_set,1)[0][:,None]
            # enc_inputs = enc_inputs / torch.max(enc_inputs,1)[0][:,None]
        else:
            enc_inputs = pic_set
        enc_inputs = enc_inputs.reshape((N*BS,)+pshape)

        if self.two_outputs:

            # encode
            h,skip = self.encoder(enc_inputs)
            # print("h")
            # print(h.min(),h.max(),h.mean())
            # print("skip")
            # for i in range(len(skip)):
            #     print("i = {}".format(i))
            #     print(skip[i].min(),skip[i].max(),skip[i].mean())

            # aggregate
            if not self.encoder.training:
                h = h.detach()
                skip = [x.detach() for x in skip]
            agg_h,agg_skip = self.aggregate(h,skip,N,BS)

            # decode
            input_i = [agg_h,agg_skip]
            dec_pics = self.decoder(input_i)
            dec_pics = dec_pics.reshape((N,BS,) + pshape)

        else:

            # encode
            h = self.encoder(enc_inputs)

            # aggregate
            if not self.encoder.training:
                h = h.detach()
            agg_h = self.aggregate(h,None,N,BS)
        
            # decode
            input_i = agg_h
            dec_pics = self.decoder(input_i)
            dec_pics = dec_pics.reshape((N,BS,) + pshape)


        # projector
        # proj = self.projector(h).reshape(N,BS,-1)

        return dec_pics,h

    def aggregate(self,h,skip,N,BS):
        agg_fxn = self._agg_fxn
        agg_type = self._agg_type
        if agg_fxn == 'mean':
            return share_encoding_mean(agg_type,h,skip,N,BS)
        elif agg_fxn == 'id':
            return h,skip
        else:
            raise ValueError(f"Uknown aggregation function [{agg_fxn}]")
