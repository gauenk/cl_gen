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
from layers.unet import SingleConv

def score_cog(cfg,image_volume,backbone,nn_params,train_steps):

    """
    [example]:

    backbone = UNet_small
    nn_params = {'lr':1e-3}
    train_steps = 1000
    score_cog(cfg,image_volume,backbone,nn_params,train_steps)

    """

    cog = COG(backbone,cfg.nframes,cfg.device,nn_params=nn_params,train_steps=train_steps)
    cog.train_models_mp(image_volume)
    recs = cog.test_models(image_volume).detach().to(cfg.device) # no grad to unets
    # psnrs = compute_recs_psnrs(recs,clean).to(cfg.device).reshape(P,-1)
    # cog.logit = logit
    # score = cog.logit(psnrs/160.)
    # score = cog.logit(recs)
    # score = cog.logit_consistency_score(recs,image_volume)
    features = cog.consistency_score(recs,image_volume)
    print("features")
    print(features)
    score = features[1,0].item()
    print("score",score)
    return score

class AlignmentDetector(nn.Module):
    def __init__(self, nframes,color=3):
        super(AlignmentDetector, self).__init__()
        self.conv1 = SingleConv(color*(nframes-1)*nframes, 32,
                                kernel_size=2, padding=1, stride=1)
        self.conv2 = SingleConv(32, 64, 1)
        self.conv3 = SingleConv(64, 128, 1)
        self.conv4 = SingleConv(128, 128, 1)
        self.linear = nn.Linear(128*2*2,nframes)
        self.s = nn.Sigmoid()

    def forward(self, x):
        B = x.shape[0]
        x = rearrange(x,'b m t c h w -> b (m t c) h w')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(B,-1)
        x = self.linear(x)
        return self.s(x)

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
        self.proc_limit = 5
        self.nn_backbone = nn_backbone
        self.device = device
        self.nframes = nframes
        if nn_params is None: self.nn_params = edict({'lr':1e-3})
        else: self.nn_params = nn_params
        self.train_steps = train_steps
        self.models = []
        self.optims = []
        self._init_learning(nframes)

        input_dim = 20
        self.logit = Logistic(input_dim,nframes).to(device)
        # self.logit = AlignmentDetector(nframes).to(device)
        self.idx = -1

    def reset(self):
        self._init_learning(self,nframes)

    def _init_learning(self,nframes):
        nn_params = self.nn_params
        unetp = self.nn_params.init_params
        self.models,self.optims = [],[]
        for t in range(nframes):
            if t == nframes//2:
                self.models.append(None)
                self.optims.append(None)
                continue
            torch.manual_seed(123)
            model = self.nn_backbone(3).to(self.device)
            if t > 0 and (unetp is None): model.load_state_dict(self.models[0].state_dict())
            elif not(unetp is None): model.load_state_dict(unetp.state_dict())
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
            if len(procs) == self.proc_limit: self.finish_procs(procs)
        self.finish_procs(procs)
                
    def finish_procs(self,procs):
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
                burst_tb = rearrange(burst,'t b c h w -> (t b) c h w')
                recs_i = self.models[t_i](burst_tb).detach()
                recs_i_rs = rearrange(recs_i,'(t b) c h w -> t b c h w',t=T)
                recs.append(recs_i_rs)
            recs = torch.stack(recs,dim=0)
            recs = rearrange(recs,'tm1 t b c h w -> b tm1 t c h w')
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
        B = recs.shape[0]
        Tm1,T = recs.shape[1:3]
        c1,c2 = cmpr.shape[1:3]
        simmat = torch.zeros(B,Tm1,c1,T,c2)
        for b in range(B):
            for m_i in range(Tm1):
                for m_j in range(c1):
                    for l in range(T):
                        for k in range(c2):
                            #simmat[m_i,m_j,l,k] = 1000*F.mse_loss(recs[m_i,l],recs[m_j,k]).item()
                            psnrs = images_to_psnrs(recs[b,m_i,l],cmpr[b,m_j,k])
                            simmat[b,m_i,m_j,l,k] = np.mean(psnrs)
        return simmat

    def _pixel_shuffle_uniform(self,burst,R):
        T,C,H,W = burst.shape
        S = H*W
        indices = torch.randint(0,T,(R,S),device=burst.device)
        shuffle = torch.zeros((C,R,S),device=burst.device)
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
        S = H*W
        order = torch.stack([torch.randperm(T,device=burst.device) for _ in range(S)],dim=1)
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
        device = burst.device
        T,B,C,H,W = burst.shape
        R = T
        noisy = torch.zeros((B,R,C,H,W),device=device)
        target = torch.zeros((B,R,C,H,W),device=device)
        for b in range(B):
            noisy[b] = burst[:,b]#self._pixel_shuffle_uniform(burst[:,b],R)
            target[b] = self._pixel_shuffle_uniform(burst[:,b],R)
        noisy = rearrange(noisy,'b r c h w -> (b r) c h w')
        target = rearrange(target,'b r c h w -> (b r) c h w')
            
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

    def consistency_score(self,recs,noisy):
        # -- sizes --
        B = recs.shape[0]

        # -- matrices --
        noisy_rs = rearrange(noisy,'t b c h w -> b 1 t c h w')
        mat_rr = self.compute_consistency_mat(recs,recs)
        mat_rn = self.compute_consistency_mat(recs,noisy_rs)
        mat_nn = self.compute_consistency_mat(noisy_rs,noisy_rs)

        # -- v1 features of matrices --
        ftrs_rr = self.mat_fxn_v1(mat_rr).reshape(B,-1,3)
        ftrs_rn = self.mat_fxn_v1(mat_rn).reshape(B,-1,3)
        ftrs_nn = self.mat_fxn_v1(mat_nn).reshape(B,-1,3)
        # full_features = torch.cat([ftrs_rr,ftrs_rn,ftrs_nn],dim=1)
        m_rr = torch.mean(ftrs_rr,dim=(0,1))
        m_rn = torch.mean(ftrs_rn,dim=(0,1))
        m_nn = torch.mean(ftrs_nn,dim=(0,1))
        features = torch.stack([m_rr,m_rn,m_nn],dim=0)
        features = features.to(self.device)
        return features

    def logit_consistency_score(self,recs,noisy):
        features = self.consistency_score(recs,noisy)
        features = features.reshape(1,-1)
        scores = self.logit(features)
        return scores
        
    def mat_fxn_v1(self,mat):
        B,A1,A2,B1,B2 = mat.shape
        identity = torch.eye(B1,B2)
        nframes = B1
        assert B1 == B2,"Are these always the nframes?"
        features = []
        for b in range(B):
            for a1 in range(A1):
                for a2 in range(A2):

                    # -- index element --
                    elem = mat[b,a1,a2]

                    # -- diag diff --
                    diag = torch.diag(elem)
                    diag_diff = torch.abs(torch.mean(diag - diag[nframes//2]))

                    # -- offset extrema --
                    min_diag = ( 1 - identity ) * elem
                    max_diag = min_diag + identity * 160.
                    max_offset = torch.max(min_diag)/160.
                    min_offset = torch.min(max_diag)/160.
                    ftrs = torch.FloatTensor([diag_diff,max_offset,min_offset])
                    features.append(ftrs)
        features = torch.stack(features,dim=0)
        features = rearrange(features,'(b a1 a2) f -> b a1 a2 f',b=B,a1=A1)
        return features

    def operator_consistency(self,recs,noisy):
        T = recs.shape[1]
        noisy_rs = rearrange(noisy,'t b c h w -> b 1 t c h w')
        mat_rr = self.compute_consistency_mat(recs,recs)
        mat_rn = self.compute_consistency_mat(recs,noisy_rs)
        mat_nn = self.compute_consistency_mat(noisy_rs,noisy_rs)

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
