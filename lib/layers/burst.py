""" Full assembly of the parts to form the complete network """



# -- python imports --
import numpy as np
import numpy.random as npr
from einops import rearrange, repeat, reduce


# -- pytorch imports --
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- project imports --
# from layers.kpn.KPN import LossFunc as kpnLossFunc
# from layers.kpn.KPN import LossBasic,LossAnneal,TensorGradient
from layers.ot_pytorch import sink_stabilized,sink,pairwise_distances

class BurstAlign(nn.Module):

    def __init__(self, kpn, unet_info, noise_critic, skip_middle = True):
        super(BurstAlign, self).__init__()
        self.kpn = kpn
        self.unet_info = unet_info
        self.noise_critic = noise_critic
        self.skip_middle = skip_middle
        self.global_step = 0
        self.one = torch.FloatTensor([1.]).to(self.noise_critic.one.device)
        # self.ot_loss = LossOT()

    def forward(self, burst):
        """
        :param burst: burst[:,1] ~ burst[:,N], shape: [N, batch, 3, height, width]
        """
        # -- init --
        CRITIC_UD = 1
        N,B,C,H,W = burst.shape
        kpn_stack = rearrange(burst,'n b c h w -> b n c h w')
        kpn_cat = rearrange(burst,'n b c h w -> b (n c) h w')

        # -- align images --
        aligned,aligned_ave = self.kpn(kpn_cat,kpn_stack)

        # -- denoise --
        denoised = torch.stack([self.unet_info.model(aligned[:,i]) for i in range(N)],dim=0)
        rec = torch.mean(denoised,dim=0)

        # -- no error back to unet --
        rec = rec.detach()

        # -- critic loss for noise --
        residual = aligned - rec.unsqueeze(1).repeat(1,N,1,1,1)
        if self.training and (self.global_step % CRITIC_UD) == 0: self.noise_critic.update_disc(residual)

        # -- update unet --
        if self.training: self.update_unet(aligned,residual)

        # -- update global step --
        self.global_step += 1

        # -- return all components --
        return aligned,aligned_ave,rec

    def update_unet(self, attached_aligned, residual):

        #
        # -- no prop to alignment --
        #

        aligned = attached_aligned.detach()
        # DETACH_STEP = 300
        # if self.global_step < DETACH_STEP:
        #     aligned = attached_aligned.detach()
        # else:
        #     aligned = attached_aligned

        # aligned_ave = torch.mean(aligned, dim=1)
        # ot_loss = self.ot_loss(aligned,aligned_ave).item()
        # if ot_loss > 0.01: return

        #
        # -- init --
        #

        model = self.unet_info.model
        optim = self.unet_info.optim
        S = self.unet_info.S
        B,N,C,H,W = aligned.shape

        #
        # -- check if the frames were actually aligned; pos = real, neg = fake --
        #

        raw_score = self.noise_critic.compute_fake_loss(residual,False).detach()
        zo_coeff = torch.sigmoid(raw_score).detach()
        zo_coeff = rearrange(zo_coeff,'(b n) -> b n',b=B)
        
        #
        # -- train n2n model --
        #
        
        pairs,S = self._select_pairs(S,N)
        for (i,j) in pairs:
            # -- zero grad --
            model.zero_grad()
            optim.zero_grad()

            # -- forward pass --
            ai,aj = aligned[:,i],aligned[:,j]
            pred = model(ai)

            # -- compute loss --
            loss = torch.mean(F.mse_loss(pred,aj,reduction='none'),(1,2,3))
            
            # -- compute lr coeff --
            lr_coeff = torch.min(torch.stack([zo_coeff[:,i],zo_coeff[:,j]],dim=0),dim=0)[0]

            # -- optim step --
            # loss.backward(lr_coeff)
            torch.mean(loss).backward()
            # if self.global_step < DETACH_STEP:
            #     loss.backward()
            # else:
            #     loss.backward(retain_graph=True)
            optim.step()

    def denoise(self, x):
        recs,rec = self.kpn(x,x)
        self.unet(recs[i] for i in range(N))
        
    def _select_pairs(self, S, N):
        if self.skip_middle:
            pairs = list(set([(i,j) for i in range(N) for j in range(N) if i != N//2 and j != N//2]))
        else:
            pairs = list(set([(i,j) for i in range(N) for j in range(N)]))
        P = len(pairs)
        if S is None: S = P
        r_idx = npr.choice(range(P),S)
        s_pairs = [pairs[idx] for idx in r_idx]
        return s_pairs,S
        

"""
Loss Functions

"""

class AlignLoss(nn.Module):
    """
    loss function of KPN
    """
    def __init__(self, noise_critic, coeff_ave=1.0, coeff_burst=1.0, coeff_ot=100.0, gradient_L1=True,
                 alpha=0.9998, beta=0.9772, reg=0.5, S=None):
        super(AlignLoss, self).__init__()
        self.noise_critic = noise_critic
        self.coeff_ave = coeff_ave
        self.coeff_burst = coeff_burst
        self.coeff_ot = coeff_ot
        self.alpha,self.beta = alpha,beta

        use_tensor_grad = False # experiment using unsup, blind, no OT, rec_img for losses (no gt_img), noise critic, [with & without tensor_grad] show 23.46 v 25.71 PSNR
        self.loss_rec = LossRec(gradient_L1,use_tensor_grad)
        self.loss_burst = LossRecBurst(gradient_L1,use_tensor_grad)
        self.loss_ot = LossOT(reg,S)

    def forward(self, aligned, aligned_ave, rec_img, ground_truth, sm_raw_img, global_step):
        """
        forward function of loss_func
        :param aligned: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param aligned_ave: \frac{1}{N} \sum_i^N frame_i, shape: [batch, 3, height, width]
        :param rec_img: \frac{1}{N} \sum_i^N denoised(frame_i), shape: [batch, 3, height, width]
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        # -- init --
        CRITIC_UD = 10
        B,N,C,H,W = aligned.shape

        # -- decaying coefficients --
        coeff_anneal_f = self._anneal_fast_coeff(global_step)
        if coeff_anneal_f < 0.1: coeff_anneal_f = 0.1
        coeff_anneal_s = self._anneal_slow_coeff(global_step)

        # -- alignment averages loss via MSE --
        loss_ave = self.loss_rec(aligned_ave, rec_img)
        # loss_ave = self.loss_rec(aligned_ave, rec_img)
        # loss_ave = coeff_anneal_f * coeff_anneal_s * self.coeff_ave * self.loss_rec(aligned_ave, ground_truth)
        # loss_ave = self.coeff_ave * self.loss_rec(aligned_ave, ground_truth)

        # -- alignment loss for each frame via MSE --
        loss_burst = self.coeff_burst * self.loss_burst(aligned, rec_img)
        
        # -- noise pattern loss using sinkhorn --
        # loss_ot = self.coeff_ot * self.loss_ot(aligned, ground_truth)
        # loss_ot = self.coeff_ot * self.loss_ot(aligned, rec_img)
        loss_ot = 0 * self.loss_rec(aligned_ave, rec_img)

        # -- noise pattern loss using critic --
        residual = aligned - sm_raw_img.unsqueeze(1).repeat(1,N,1,1,1)
        loss_nc = 0 * self.noise_critic.compute_fake_loss(residual)
        if (global_step % CRITIC_UD) != 0: loss_nc *= 0

        return loss_nc, loss_ave, loss_burst, loss_ot

    def _anneal_slow_coeff(self, global_step):
        return self.alpha ** global_step        

    def _anneal_fast_coeff(self, global_step):
        return self.beta ** global_step        

class LossRec(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True, tensor_grad=True):
        super(LossRec, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)
        self.tensor_grad = tensor_grad

    def forward(self, pred, ground_truth):
        loss = self.l2_loss(pred, ground_truth)
        if self.tensor_grad:
            loss += self.l1_loss(self.gradient(pred), self.gradient(ground_truth))
        return loss
               

class LossRecBurst(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, gradient_L1=True,tensor_grad=True):
        super(LossRecBurst, self).__init__()
        self.global_step = 0
        self.loss_rec = LossRec(gradient_L1=gradient_L1,tensor_grad=tensor_grad)

    def forward(self, pred_i, ground_truth):
        """
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_rec(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.size(1)
        return loss


class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )

class LossOT(nn.Module):

    def __init__(self,reg=0.5,S=None,skip_middle=False):
        super(LossOT,self).__init__()
        self.reg = reg
        self.S = 10
        self.skip_middle = skip_middle

    def forward(self,burst,gt_img):
        """
        :param burst: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param gt_img: shape [batch, 3, height, width]
        """
        BS,N,C,H,W = burst.shape
        S = self.S

        r_gt_img = gt_img.unsqueeze(1).repeat(1,N,1,1,1)
        diffs = r_gt_img - burst
        diffs = rearrange(diffs,'b n c h w -> b n (h w) c')
        
        ot_loss = 0
        for b in range(BS):
            pairs,S = self._select_pairs(S,N)
            for (i,j) in pairs:
                di,dj = diffs[b,i],diffs[b,j]
                M = torch.sum(torch.pow(di.unsqueeze(1) - dj,2),dim=-1)
                ot_loss += sink_stabilized(M, self.reg)
        ot_loss /= S*BS
        return ot_loss

        
    def _select_pairs(self,S,N):
        if self.skip_middle:
            pairs = list(set([(i,j) for i in range(N) for j in range(N) if i<j and i != N//2 and j != N//2]))
        else:
            pairs = list(set([(i,j) for i in range(N) for j in range(N) if i<j]))
        P = len(pairs)
        if S is None: S = P
        r_idx = npr.choice(range(P),S)
        s_pairs = [pairs[idx] for idx in r_idx]        
        return s_pairs,S
        

class NoiseCriticModel():

    def __init__(self,disc,optim,sim_params,device):
        self.disc = disc
        self.optim = optim
        self.sim_params = sim_params
        self.device = device
        self.one = torch.FloatTensor([1.]).to(device)
        self.mone= -1 * self.one

    def compute_fake_loss(self,fake,use_mean=True):
        # -- reshape fake data --
        fake = rearrange(fake,'b n c h w -> (b n) c h w')

        # -- init --
        one = self.one

        # -- compute loss --
        output = self.disc(fake).view(-1)
        if use_mean: error_gen = output.mean(0).view(1)
        else: error_gen = output

        # -- steps done outside this function --
        # error_gen.backward(one)
        # optimizer_gen.step()

        return error_gen

    def update_disc(self,fake,real=None):
        """
        Maximize Discriminator
        """
        # -- reshape fake data --
        fake = rearrange(fake,'b n c h w -> (b n) c h w')

        # -- init info --
        B,C,H,W = fake.shape
        one,mone = self.one,self.mone
        if real is None: real = self.sim_noise(fake.shape)
        
        # -- (i) real samples --
        output = self.disc(real).view(-1)
        err_disc_real = output.mean(0).view(1)
        err_disc_real.backward(one)
        D_x = output.mean().item()

        # -- (ii) fake samples --
        output = self.disc(fake.detach()).view(-1)
        err_disc_fake = output.mean(0).view(1)
        err_disc_fake.backward(mone)
        D_G_z1 = output.mean().item()

        # -- compute difference --
        error_disc = err_disc_real - err_disc_fake
        self.optim.step()

        # -- simplify function (WGan) --
        for p in self.disc.parameters():
            p.data.clamp_(-0.01, 0.01)
        
    def sim_noise(self,shape):
        if self.sim_params.noise_type == "gaussian":
            return self.sim_gaussian(shape)
        else:
            raise NotImplemented(f"Uknown noise type {self.sim_params.noise_type}")

    def sim_gaussian(self,shape):
        samples = torch.normal(self.sim_params.mean,self.sim_params.std * torch.ones(shape))
        samples = samples.to(self.device)
        return samples


