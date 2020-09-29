
""" 
Create the network to disentangle dynamic attributes from static information
 """

# numpy imports
import numpy as np

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClBlockLoss(nn.Module):

    def __init__(self,hyperparams, num_transforms, batch_size):
        super(ClBlockLoss, self).__init__()
        self.hyperparams = hyperparams
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.num_transforms = num_transforms
        self.batch_size = batch_size

        msizes = [2,self.num_transforms,self.num_transforms+1]
        self.masks_neg,self.masks_pos = self.get_masks(msizes,batch_size)
        

    def forward(self,z):
        
        # shape info
        N,BS = z.shape[:2]

        # reshape
        z = z.reshape(N,BS,-1)

        # compute CL loss
        loss = self.compute_loss_simclr(z)

        return loss
        
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


