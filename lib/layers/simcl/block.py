"""
Generalized contrastive learning loss
"""

# pytorch imports
import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F

from ..identity import Identity
from torchvision.models.resnet import ResNet

class ClBlock(nn.Module):

    def __init__(self,encoder,projector,device,
                 num_transforms,batch_size):
        super(ClBlock, self).__init__()
        self.encoder = encoder
        self.projector = projector
        
        # n_features = 2048
        # projection_dim = 64
        # self.projector = nn.Sequential(
        #     nn.Linear(n_features, n_features, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(n_features, projection_dim, bias=False),
        # )

        self.device = device
        self.num_transforms = num_transforms
        self.batch_size = batch_size
        
        self.return_three = False # generally this is always False
        self.two_outputs = True
        if isinstance(self.encoder,ResNet):
            # self.two_outputs = False
            self.encoder.fc = Identity()
        else:
            self.two_outputs = True

    def forward(self,pic_set):

        # get shape info
        N = len(pic_set)
        BS = pic_set[0].shape[0]
        pshape = pic_set[0][0].shape
        shape = (N,BS,) + pshape

        # encode
        pic_set = pic_set.reshape((N*BS,)+pshape)
        skips = None
        if self.two_outputs:
            h,skips = self.encoder(pic_set)
        else:
            h = self.encoder(pic_set)

        # project
        proj = self.projector(h).reshape(N,BS,-1)
        
        h = h.reshape(N,BS,-1)
        if self.return_three:
            return h,proj,skips
        else:
            return h,proj

class ClBlockEnc(nn.Module):

    def __init__(self,model,img_size,d_model_attn):
        super(ClBlockEnc, self).__init__()
        self.model = model
        self.model.two_outputs = True
        self.model.return_three = True
        self.img_size = img_size
        self.d_model_attn = d_model_attn
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,pic_set):
        img_size = self.img_size
        h,p,skips = self.model(pic_set)
        N,BS,E = h.shape
        h = h.view(N,BS,-1,img_size,img_size)
        agg = [h]
        for skip in skips[1:]:
            if skip.shape[-1] != self.img_size:
                skip = self.up(skip)
            skip = skip.view(N,BS,-1,img_size,img_size)
            # print(skip.shape)
            agg.append(skip)
        rep = torch.cat(agg,dim=2)
        rep = rep[:,:,:self.d_model_attn]
        return rep


