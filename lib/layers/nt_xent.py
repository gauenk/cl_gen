import torch
import torch.nn as nn
import torch.distributed as dist


class NT_Xent(nn.Module):

    def __init__(self, batch_size, temperature, device, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def mask_sim(self, num_transforms, batch_size):
        K = num_transforms * batch_size
        mask = torch.ones((K,K), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            for n in range(num_transforms):
                mask[i, n*batch_size + i] = 0
                mask[n*batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        # no world size > 1
        # if self.world_size > 1:
        #     z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        print(sim)
        print(sim.numel())
        sim = torch.arange(0,sim.numel()).reshape(sim.shape)
        # these are the "same" diags
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )

        negative_samples = sim[self.mask].reshape(N, -1)
        print(negative_samples.shape)
        # positive_samples = torch.arange(0,N).reshape(N,1)
        # negative_samples = torch.arange(100,100+N*8).reshape(N,8)

        self.batch_size = 2
        num_transforms = 3
        N = num_transforms*self.batch_size
        K = (num_transforms-1)*num_transforms*self.batch_size
        mask_neg = self.get_mask_sim(num_transforms,self.batch_size)
        mask_pos = self.get_mask_diff(mask_neg)
        simmat = torch.arange(0,N**2).reshape(N,N)
        print(simmat)
        print('simmat.shape',simmat.shape)
        print(mask_pos.shape)
        print(mask_pos.type(torch.int))
        print(K)
        pos_samples = simmat[mask_pos].reshape(K,1) # NumA x 1
        neg_samples = simmat[mask_neg].reshape(N,-1) # NumA x NumA-2 ("same" and "1")
        logits = []
        for n in range(num_transforms-1):
            logit = torch.cat((pos_samples[n::2],neg_samples),dim=1)
            logits.append(logit)
        logits = torch.cat(logits,dim=0)
        labels = torch.zeros(K).to(pos_samples.device).long()
        print(logits.shape)
        print(logits)
        print('psn',pos_samples.numel())
        # logits = torch.cat((pos_samples, neg_samples), dim=1)
        print(logits)
        print(labels)
        exit()

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        print(logits)
        print(logits.shape)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def get_mask_sim(self, num_transforms, batch_size):
        import numpy as np
        K = num_transforms * batch_size
        mask = np.zeros((K,K), dtype=np.int)
        print(num_transforms)
        print(batch_size)
        for n in range(num_transforms-1):
            ones = np.ones(K-batch_size*(n+1))
            dmask = np.diag(ones,(n+1)*batch_size).astype(np.int)
            mask += dmask
            dmask = np.diag(ones,-(n+1)*batch_size).astype(np.int)
            mask += dmask
        # mask[1::2,1::2] = 0
        mask = np.logical_not(mask)
        np.fill_diagonal(mask,0)
        print(mask.astype(np.int))
        mask = torch.from_numpy(mask).type(torch.bool)
        return mask
        # K = num_transforms * batch_size
        # mask = torch.ones((K,K), dtype=bool)
        # mask = mask.fill_diagonal_(0)
        # for i in range(batch_size):
        #     for n in range(num_transforms):
        #         mask[i, n*batch_size + i] = 0
        #         mask[n*batch_size + i, i] = 0
        # print(mask.type(torch.int))
    
        # return mask

    def get_mask_diff(self,mask_sim):
        mask = mask_sim.clone()
        mask = torch.logical_not(mask)
        mask = mask.fill_diagonal_(0)
        return mask

