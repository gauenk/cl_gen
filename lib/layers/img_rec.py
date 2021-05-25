
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from pyutils import add_noise

from .identity import Identity


class ImgRecLoss(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, decoder, encoder, projector,
                 blind=False, gamma = 0.05, beta = 0.1):
        super(ImgRecLoss, self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.projector = projector
        self.blind = blind # doesn't do anything; need to
        self.gamma = gamma
        self.beta = beta

    def forward(self, reconstructed_pics, target_projections, transforms):
        """
        pics are the estimated image

        target_projections[i][j] = projection(encoder(transform_i(image_j)))
        list (i) of tensors (j) with 
        i \in {1,\ldots,NumTransforms}
        j \in {1,\ldots,BatchSize}
        
        L_swap
        ======

        k \in {1,\ldots,M}
        i \in {1,\ldots,N}

        Given transform_i(image_k) and \{transform_j(\cdot)\}_{j=1}^N,

        target_projection_{k,i} = projection(encoder(transform_i(image_k)))
        est_projection_{k,j}    = projection(encoder(transform_j(recon_image_k)))
                    = projection(encoder(transform_j(decoder(encoder(transform_i(image_k))))))

        L_swap "=" \sum_{k=1}^M \sum_{i\neq j} loss_swap(target_project_{k,i},est_project_{k,j})
        
        L_self
        =======

        Given transform_i(image_k) and \{transform_j(\cdot)\}_{j=1}^N,

        target_projection_{k,i} = projection(encoder(transform_i(image)))
        est_projection_{k,j}    = projection(encoder(transform_i(recon_image)))
                    = projection(encoder(transform_i(decoder(encoder(transform_i(image_k))))))
        since 

        recon_image = decoder(encoder(transform_i(image_k)))
        
        Then,

        L_self "=" \sum_{k=1}^M \sum_{i=1}^N loss_self(target_project_{k,i},est_project_{k,i})


        L_proxX
        =======

        reconstructed_pics = recon_pic_{k,i}


        recon_pic_1_{k,i} = decoder(encoder(transform_i(recon_pic_{k,i})))
             = decoder(encoder(transform_i(decoder(encoder(transform_i(image_k))))))
        recon_pic_{k,i} = decoder(encoder(transform_i(image_k)))
        
        L_proxX 
        "=" 
        \sum_{k=1}^M \sum_{i=1}^N loss_proxX(recon_pic_1_{k,i},recon_pic_{k,i})


        TODO:
        - allow coefficient for L_proxX to grow as the validation accuracy improves
        
        """
        # TODO: implement EM to estimate blind transformations
        batch_size = len(transforms)
        num_transforms = len(transforms[0])

        # all pairwise comparisons; N(N-1)
        # we'd like to "batch" the encoder step.
        L_swap,L_self,L_proxX = 0,0,0
        z_list = []
        for n in range(num_transforms):
            for k in range(num_transforms):
                t_x = []
                for t in range(batch_size): # this loop allocates +5gb gpu for batchsize 64
                    rpic = reconstructed_pics[n][t]
                    trans_k = transforms[t][k]
                    t_x.append(add_noise(trans_k,rpic))
                t_x = torch.stack(t_x)
                # print("t_x.shape",t_x.shape)
                h_ij = self.encoder(t_x)
                # print("h_ij.shape",h_ij.shape)
                z_ij = self.projector(h_ij)
                if n == k:
                    L_self += F.mse_loss(z_ij,target_projections[n])
                    h_ji = h_ij.detach()
                    pic_hat = self.decoder([h_ij])[0]
                    # print(pic_hat.shape,reconstructed_pics[n].shape)
                    L_proxX += F.mse_loss(pic_hat,reconstructed_pics[n])
                else:
                    L_swap += F.mse_loss(z_ij,target_projections[n])

        L = L_swap + self.gamma * L_self + self.beta * L_proxX
        L /= len(reconstructed_pics)
        return L
