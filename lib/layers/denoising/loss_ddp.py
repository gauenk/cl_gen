""" 
Create the network to disentangle dynamic attributes from static information
 """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# local import
from .utils import share_encoding_mean,share_encoding_mean_check

class DenoisingLossDDP(nn.Module):

    def __init__(self,hyperparams, num_transforms, batch_size,
                 img_loss_type='l2',enc_loss_type='simclr'):
        super(DenoisingLossDDP, self).__init__()
        self.hyperparams = hyperparams
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.num_transforms = num_transforms
        self.batch_size = batch_size

        msizes = [2,self.num_transforms,self.num_transforms+1]
        self.masks_neg,self.masks_pos = self.get_masks(msizes,batch_size)
        self.img_loss_type = img_loss_type
        self.enc_loss_type = enc_loss_type
        

    def forward(self,pic_set,dec_pics,h):
        hyperparams = self.hyperparams
        simh = []
        simpics = []

        N = len(pic_set)
        BS = pic_set[0].shape[0]
        pshape = pic_set[0][0].shape
        shape = (N,BS,) + pshape

        h = h.reshape(N,BS,-1)
        loss_h = self.compute_enc_loss(h)

        # pair-wise losses
        pic_set = pic_set.reshape(N,BS,-1)
        dec_pics = dec_pics.reshape(N,BS,-1)
        offset_idx = [(i+1)%N for i in range(N)]
        pic_pair = [pic_set,dec_pics[offset_idx]]
        loss_pairs = self.compute_img_loss(pic_pair)

        # across decoded pics
        # dec_pics = [dec_pic for dec_pic in dec_pics]
        # loss_x = self.compute_img_loss(dec_pics)
        # + hyperparams.x * loss_x

        loss = loss_pairs + hyperparams.h * loss_h
        return loss
    
    def aggregate(self,h,aux,N,BS):
        agg_fxn = self._agg_fxn
        agg_type = self._agg_type
        if agg_fxn == 'mean':
            return share_encoding_mean(agg_type,h,aux,N,BS)
        elif agg_fxn == 'id':
            return h,aux
        else:
            raise ValueError(f"Uknown aggregation function [{agg_fxn}]")

    def compute_img_loss(self,sim_i):
        if self.img_loss_type == 'simclr':
            return self.compute_loss_simclr(sim_i)
        elif self.img_loss_type == 'l2':
            return F.mse_loss(sim_i[0],sim_i[1])
        else:
            raise ValueError(f"Unknown img loss type [{self.img_loss_type}]")
        
    def compute_enc_loss(self,sim_i):
        if self.enc_loss_type == 'simclr':
            return self.compute_loss_simclr(sim_i)
        elif self.enc_loss_type == 'l2':
            return F.mse_loss(sim_i[0],sim_i[1])
        else:
            raise ValueError(f"Unknown loss type [{self.enc_loss_type}]")

    def compute_loss_simclr(self,sim_i):
        N = len(sim_i)
        BS = self.batch_size
        Kpos = N*(N-1)*BS
        Kneg = N * BS
        mask_pos = self.masks_pos[N]
        mask_neg = self.masks_neg[N]
        return self.generalized_nt_xent(sim_i,N,BS,Kpos,Kneg,mask_pos,mask_neg)


    def generalized_nt_xent(self,sim_i,N,BS,Kpos,Kneg,mask_pos,mask_neg):
        """
        sim_i [ NumTransforms x BatchSize x EncoderD]
        """
        temperature = self.hyperparams.temperature

        # compute similarity scores
        if isinstance(sim_i,list):
            s = torch.cat(sim_i,dim=0)
        elif isinstance(sim_i,torch.Tensor):
            s = sim_i.reshape(N*BS,-1)
        else:
            raise TypeError("Unknown sim_i type [{}]".format(type(sim_i)))
        simmat = self.similarity_f(s.unsqueeze(1),s.unsqueeze(0)) / temperature
        pos_samples = simmat[mask_pos].reshape(Kpos,1) # NumA x 1
        neg_samples = simmat[mask_neg].reshape(Kneg,-1) # NumA x NumA-2 ("same" and "1")

        # create logits and labels
        logits = []
        for n in range(N-1):
            logit = torch.cat((pos_samples[n::(N-1)],neg_samples),dim=1)
            logits.append(logit)
        logits = torch.cat(logits,dim=0)
        labels = torch.zeros(Kpos).to(pos_samples.device).long()

        # run loss
        loss = self.criterion(logits, labels)
        loss /= Kpos
        return loss
        

    def get_masks(self, Nsizes, BS):
        masks_neg,masks_pos = {},{}
        for N in Nsizes:
            mask_neg = self.get_mask_neg(N, BS)
            masks_neg[N] = mask_neg
            masks_pos[N] = self.get_mask_pos(mask_neg)
        return masks_neg,masks_pos

    def get_mask_neg(self, num_transforms, batch_size):
        K = num_transforms * batch_size
        mask = np.zeros((K,K), dtype=np.int)
        for n in range(num_transforms-1):
            ones = np.ones(K-batch_size*(n+1))
            dmask = np.diag(ones,(n+1)*batch_size).astype(np.int)
            mask += dmask
            dmask = np.diag(ones,-(n+1)*batch_size).astype(np.int)
            mask += dmask
        mask = np.logical_not(mask)
        np.fill_diagonal(mask,0)
        mask = torch.from_numpy(mask).type(torch.bool)
        return mask

    def get_mask_pos(self,mask_sim):
        mask = mask_sim.clone()
        mask = torch.logical_not(mask)
        mask = mask.fill_diagonal_(0)
        return mask


