""" 
Create the network to disentangle dynamic attributes from static information
 """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DisentangleLoss(nn.Module):

    def __init__(self, models, hyperparams, num_transforms, batch_size, device):
        super(DisentangleLoss, self).__init__()
        self.encoder_c = models.enc_c
        self.encoder_d = models.enc_d
        self.decoder = models.dec
        self.projector = models.proj
        self.hyperparams = hyperparams
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.device = device
        self.enc_d_size = list(self.encoder_d.modules())[-1].out_features
        
        self.num_transforms = num_transforms
        self.batch_size = batch_size

        msizes = [self.num_transforms,self.num_transforms+1]
        self.masks_neg,self.masks_pos = self.get_masks(msizes,batch_size)


    def forward(self,pic_set):
        hyperparams = self.hyperparams
        loss_repr = 0
        simh = []
        simpics = []
        zeros = torch.zeros(self.batch_size,self.enc_d_size).to(self.device)
        for i,pic_i in enumerate(pic_set):
            h_i = self.encoder_c(pic_i)
            g_i = self.encoder_d(pic_i)

            clean_enc = torch.cat([h_i,zeros],dim=1)
            clean_pic = self.decoder(clean_enc)
            simpics.append(torch.flatten(clean_pic,1))
            simh.append(h_i)

            h_sim_i = [h_i]
            g_sim_i = [g_i]
            noisypics_i = [torch.flatten(pic_i,1)]
            x_sim_i = []
            for j,pic_j in enumerate(pic_set):

                # static_i and dyanmic_j
                g_j = self.encoder_d(pic_j)
                d_ij = torch.cat([h_i,g_j],dim=1)
                x_hat_gij = self.decoder(d_ij)
                # h_hat_gij = self.encoder_c(x_hat_gij)
                g_hat_gij = self.encoder_d(x_hat_gij)
        
                # static_j and dyanmic_i
                h_j = self.encoder_c(pic_j)
                d_ji = torch.cat([h_j,g_i],dim=1)
                x_hat_hij = self.decoder(d_ji)
                h_hat_hij = self.encoder_c(x_hat_hij)
                # g_hat_hij = self.encoder_d(x_hat_gij)                

                g_sim_i.append(g_hat_gij)
                h_sim_i.append(h_hat_hij)
                noisypics_i.append(torch.flatten(x_hat_hij,1))
            # g_sim_i [NumTransforms x BatchSize x EncoderD]
            # x_sim_i [3 x BatchSize x ImgSize]
            # h_sim [NumTransforms x (BS x NumTrans) x EncoderC]
            #       [ Same            Different       Repr ]
            # x_sim [NumTransforms x 3 x BatchSize x ImgSize]
            # x_sim.append(x_sim_i)
            loss_g = self.compute_loss(g_sim_i,proj=True)
            loss_h = self.compute_loss(h_sim_i,proj=True)
            loss_x = self.compute_loss(noisypics_i,proj=False)
            loss_i = hyperparams.g * loss_g + hyperparams.h * loss_h + hyperparams.x * loss_x
            loss_repr += loss_i
        loss_x = self.compute_loss(simpics,proj=False)
        loss_h = self.compute_loss(simh,proj=True)
        loss = loss_repr / len(pic_set) + hyperparams.x * loss_x + hyperparams.h * loss_h
        return loss


    def compute_loss(self,sim_i,proj=True):
        N = len(sim_i)
        BS = self.batch_size
        Kpos = N*(N-1)*BS
        Kneg = N * BS
        mask_pos = self.masks_pos[N]
        mask_neg = self.masks_neg[N]
        if proj: sim_i = [self.projector(x) for x in sim_i]
        return self.generalized_nt_xent(sim_i,N,BS,Kpos,Kneg,mask_pos,mask_neg)


    def generalized_nt_xent(self,sim_i,N,BS,Kpos,Kneg,mask_pos,mask_neg):
        """
        sim_i [ NumTransforms x BatchSize x EncoderD]
        """
        temperature = self.hyperparams.temperature

        # compute similarity scores
        s = torch.cat(sim_i,dim=0)
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


