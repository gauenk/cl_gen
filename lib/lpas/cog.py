"""
Noise2Noise Consistency Graph (COG)

Measures how "aligned" different frames are.

This does not (yet) tell me which ones are misaligned tho.

cog = COG(UNet_small,T,image.device,nn_params=None,train_steps=train_steps)
cog.train_models_mp(noisy)
recs = cog.test_models(noisy)
score = cog.operator_consistency(recs,noisy) # bigger = more aligned.

"""

# -- python imports --
import random
import numpy as np
import pandas as pd
from pathlib import Path
import numpy.random as npr
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp

# -- project imports --
from pyutils.misc import images_to_psnrs
from .utils import save_image

class Logistic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight,gain=1.0)
        self.s = nn.Sigmoid()

    def forward(self, x):
        return self.s(self.linear(x))
        
class COG():
    
    def __init__(self,nn_backbone,nframes,device,nn_params=None,train_steps=100):
        self.nn_backbone = nn_backbone
        self.device = device
        self.nframes = nframes
        if nn_params is None: self.nn_params = edict({'lr':1e-3})
        else: self.nn_params = nn_params
        self.train_steps = train_steps
        self.models = []
        self.optims = []
        self._init_learning(nframes)

        input_dim = 10
        self.logit = Logistic(input_dim,nframes).to(device)
        self.idx = -1

    def reset(self):
        self._init_learning(self,nframes)

    def _init_learning(self,nframes):
        nn_params = self.nn_params
        self.models,self.optims = [],[]
        for t in range(nframes):
            if t == nframes//2:
                self.models.append(None)
                self.optims.append(None)
                continue
            torch.manual_seed(123)
            model = self.nn_backbone(3).to(self.device)
            if t > 0: model.load_state_dict(self.models[0].state_dict())
            optim = torch.optim.Adam(model.parameters(),lr=nn_params.lr,betas=(0.9,0.99))
            self.models.append(model)
            self.optims.append(optim)

    def train_models(self,burst):
        self.idx = 0
        T = burst.shape[0]
        for i in range(self.train_steps):
            self.save_tr = False
            self.idx += 1
            for t in range(T):
                if t == T//2: continue
                model,optim = self.models[t],self.optims[t]
                burst_t = torch.cat([burst[:t],burst[t+1:]],dim=0)
                # burst_t = torch.stack([burst[t],burst[T//2]],dim=0)
                self._train_model_step(burst_t,model,optim)

    def train_models_mp(self,burst):
        T = burst.shape[0]
        procs = []
        for t in range(T):
            if t == T//2: continue
            burst_t = torch.cat([burst[:t],burst[t+1:]],dim=0)
            # burst_t = torch.stack([burst[t],burst[T//2]],dim=0)
            model,optim = self.models[t],self.optims[t]
            p = mp.Process(target=self.train_model_loop,
                           args=(burst,model,optim))
            p.start()
            # self.train_model_loop(burst,model,optim)
            procs.append(p)
        for p in procs:
            p.join()
        
        
    def train_model_loop(self,burst,model,optim):
        for i in range(self.train_steps):
            self._train_model_step(burst,model,optim)

    def test_models(self,burst):
        with torch.no_grad():
            T = burst.shape[0]
            recs = []
            for t_i in range(T):
                if t_i == T//2: continue
                recs_i = self.models[t_i](burst).detach()
                recs.append(recs_i)
            recs = torch.stack(recs,dim=0)
            return recs
            
    def compute_consistency(self,recs,noisy):
        T = recs.shape[1]
        simmat = self.compute_consistency_mat(recs,recs)
        simmat_rn = self.compute_consistency_mat(recs,noisy)
        simmat_nn = self.compute_consistency_mat(noisy,noisy)
        # print(simmat[0])
        # print(simmat_rn[0])
        # print(simmat_nn[0,0])
        score = 0
        for m_i in range(T-1):
            for m_j in range(T-1):
                if m_i == m_j:
                    delta = F.mse_loss(simmat[m_i,m_j,0,1],simmat[m_i,m_j,0,2]).item()
                    delta += F.mse_loss(simmat[m_i,m_j,1,0],simmat[m_i,m_j,1,2]).item()
                    delta += F.mse_loss(simmat[m_i,m_j,2,0],simmat[m_i,m_j,2,1]).item()
                else:
                    delta = self.all_pairwise_deltas(torch.diagonal(simmat[m_i,m_j]))
                    delta += self.row_pairwise_deltas_nodiag(simmat[m_i,m_j])
                score += delta
        return score

    def all_pairwise_deltas(self,vec):
        delta = 0
        for i in range(vec.shape[0]):
            for j in range(vec.shape[0]):
                delta += (vec[i] - vec[j])**2
        return delta

    def row_pairwise_deltas_nodiag(self,mat):
        delta = 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                for k in range(mat.shape[1]):
                    if j == i or k == i: continue
                    delta += F.mse_loss(mat[i,j],mat[i,k]).item()
        return delta
        
    def compute_consistency_mat(self,recs,cmpr):
        Tm1,T = recs.shape[:2]
        c1,c2 = cmpr.shape[:2]
        simmat = torch.zeros(Tm1,c1,T,c2)
        for m_i in range(Tm1):
            for m_j in range(c1):
                for l in range(T):
                    for k in range(c2):
                        #simmat[m_i,m_j,l,k] = 1000*F.mse_loss(recs[m_i,l],recs[m_j,k]).item()
                        simmat[m_i,m_j,l,k] = np.mean(images_to_psnrs(recs[m_i,l],cmpr[m_j,k]))
        return simmat

    def _pixel_shuffle_uniform(self,burst,B):
        T,C,H,W = burst.shape
        R = H*W
        indices = torch.randint(0,T,(B,R),device=burst.device)
        shuffle = torch.zeros((C,B,R),device=burst.device)
        along = torch.arange(T)
        cburst = rearrange(burst,'t c h w -> c t (h w)')
        for c in range(C):
            shuffle[c] = torch.gather(cburst[c],0,indices)
        shuffle = rearrange(shuffle,'c t (h w) -> t c h w',h=H)
        # save_image(burst,"burst.png")
        # save_image(shuffle,"shuffle.png")
        return shuffle

    def _pixel_shuffle_perm(self,burst):
        T,C,H,W = burst.shape
        R = H*W
        order = torch.stack([torch.randperm(T,device=burst.device) for _ in range(R)],dim=1)
        order = repeat(order,'b r -> b c r',c=C).long()
        cburst = rearrange(burst,'t c h w -> t c (h w)')
        target = torch.zeros_like(cburst,device=cburst.device)
        target.scatter_(0,order,cburst)
        target = rearrange(target,'t c (h w) -> t c h w',h=H)
        return target

    def _train_model_step(self,burst,model,optim):
        # -- zero --
        model.zero_grad()
        optim.zero_grad()
    
        # -- rand in and out --
        T = burst.shape[0]
        B = 10
        # order = npr.permutation(T)
        # noisy = burst[order]
        # target = burst[order]
        noisy = self._pixel_shuffle_uniform(burst,B)
        target = self._pixel_shuffle_uniform(burst,B)
            
        # -- forward --
        rec = model(noisy)
        loss = F.mse_loss(rec,target)
        if self.idx % 25 == 0 and self.save_tr:
            fn = f"train_rec_{self.idx}.png"
            print(f"Saving image to {fn}")
            self.save_tr = False
            save_image(rec,fn)

        # -- optim step --
        loss.backward()
        optim.step()

    def _compute_all_xmodel_rec_terms(self,mat_rr):
        # diags = cross model consistency
        diags,dcount = 0,0
        model_self_consistency = []
        for i in range(mat_rr.shape[0]):
            for j in range(mat_rr.shape[1]):
                if i == j:
                    diag_ii = torch.diag(mat_rr[i,j])
                    msc = torch.sum(mat_rr[i,j]) - torch.sum(diag_ii)
                    msc /= (mat_rr[i,j].numel() - len(mat_rr[i,j]))
                    model_self_consistency.append(msc)
                else:
                    diag = torch.diag(mat_rr[i,j])
                    diags += torch.mean(diag)
                    dcount += 1
        diags /= dcount
        model_self_consistency = torch.stack(model_self_consistency,dim=0)
        return diags,model_self_consistency

    def _compute_loo_xmodel_rec_terms(self,mat_rr):
        # diags = cross model consistency
        diags = []
        model_self_consistency = []
        T = mat_rr.shape[0]
        for m_idx in range(mat_rr.shape[0]):
            t_loo = m_idx if m_idx < T//2 else m_idx + 1
            diags_i = []
            for m_jdx in range(mat_rr.shape[1]):
                if m_idx == m_jdx:
                    mat_ii = mat_rr[m_idx,m_jdx]
                    diag_ii = torch.diag(mat_ii)
                    msc = torch.sum(mat_ii) - torch.sum(diag_ii)
                    msc /= (mat_ii.numel() - len(mat_ii))
                    model_self_consistency.append(msc)
                else:
                    diag = mat_rr[m_idx,m_jdx,:,t_loo]
                    diags_i.append(diag)
            diags_i = torch.stack(diags_i,dim=0)
            diags.append(diags_i)
        diags = torch.stack(diags,dim=0)
        model_self_consistency = torch.stack(model_self_consistency,dim=0)
        return diags,model_self_consistency

    def _compute_consistency_rec_noisy_terms(self,mat_rn,mat_nn):
        pass

    def logit_consistency_score(self,recs,noisy):

        T = recs.shape[1]

        # -- matrices --
        mat_rr = self.compute_consistency_mat(recs,recs)
        mat_rn = self.compute_consistency_mat(recs,noisy[None,:])
        mat_nn = self.compute_consistency_mat(noisy[None,:],noisy[None,:])

        # -- flatten --
        v_rr = mat_rr.reshape(-1)
        v_rn = mat_rn.reshape(-1)
        v_nn = mat_nn.reshape(-1)
        
        # -- cat --
        features = torch.cat([v_rr,v_rn,v_nn],dim=0)[None,:]
        features /= 160.

        # -- nn --
        # print("f",features.shape,v_rr.shape,v_rn.shape,v_nn.shape)
        features = features.to(self.device)
        # self.logit = self.logit.to(self.device)
        scores = self.logit(features)
        return scores
        

    def operator_consistency(self,recs,noisy):
        T = recs.shape[1]
        mat_rr = self.compute_consistency_mat(recs,recs)
        mat_rn = self.compute_consistency_mat(recs,noisy[None,:])
        mat_nn = self.compute_consistency_mat(noisy[None,:],noisy[None,:])

        # print("-"*10,"MAT_RR","-"*10)
        # print(mat_rr)
        # print("-"*10,"MAT_RN","-"*10)
        # print(mat_rn)
        # print("-"*10,"MAT_NN","-"*10)
        # print(mat_nn)
        

        # diags,msc = self._compute_all_xmodel_rec_terms(mat_rr)
        diags,msc = self._compute_loo_xmodel_rec_terms(mat_rr)
        # distortion = self._compute_consistency_rec_noisy_terms(mat_rn,mat_nn)
        # print(diags,msc,torch.mean(mat_rn))
        # print("DIAGS,MSC")
        # print(diags)
        # print(msc)

        score = (torch.mean(diags) + torch.mean(mat_rn)).item()/3.
        # print("SCORE")
        # print(score)

        return score
