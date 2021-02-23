""" Full assembly of the parts to form the complete network """



# -- python imports --
import numpy as np
import numpy.random as npr
from einops import rearrange, repeat, reduce

# -- pytorch imports --
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

# -- project imports --
# from layers.kpn.KPN import LossFunc as kpnLossFunc
# from layers.kpn.KPN import LossBasic,LossAnneal,TensorGradient
from layers.ot_pytorch import sink_stabilized,sink,pairwise_distances

class BurstAlignSG(nn.Module):

    def __init__(self, align_info, denoiser_info, unet_info, use_alignment = True,
                 use_unet = True, use_unet_only = False, skip_middle = True):
        super(BurstAlignSG, self).__init__()
        self.align_info = align_info
        self.denoiser_info = denoiser_info
        self.unet_info = unet_info
        self.skip_middle = skip_middle
        self.use_alignment=use_alignment
        self.use_unet = use_unet
        self.use_unet_only = use_unet_only
        self.global_step = 0
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
        mid_img_r = burst[N//2].unsqueeze(1).repeat(1,N,1,1,1)

        # -- align images --
        aligned,aligned_ave,temporal_loss,aligned_filters = self.align_info.model(kpn_stack)

        # -- denoise --
        aligned_stack = aligned
        # both_stack = torch.cat([aligned,kpn_stack],dim=2)
        # both_stack = torch.cat([mid_img_r,kpn_stack],dim=2)
        # aligned_cat = rearrange(both_stack,'b n c h w -> b (n c) h w')
        aligned_cat = rearrange(aligned_stack,'b n c h w -> b (n c) h w')
        if self.use_alignment:
            denoised,denoised_ave,denoised_filters = self.denoiser_info.model(aligned_cat,aligned_stack)
        else:
            denoised,denoised_ave,denoised_filters = self.denoiser_info.model(kpn_cat,kpn_stack)

        # -- denoised via unet --
        if self.use_unet or self.use_unet_only:
            aligned,aligned_ave,aligned_filters = denoised,denoised_ave,denoised_filters
            aligned_cat = rearrange(aligned,'b n c h w -> b (n c) h w')
            if self.use_unet_only:
                denoised = self.unet_info.model(kpn_cat)
                denoised = rearrange(denoised,'b (n c) h w -> b n c h w',n=N)
                denoised_ave = torch.mean(denoised,dim=1)
            else:
                # denoised,denoised_ave,denoised_filters = self.unet_info.model(aligned_cat,aligned)
                denoised = self.unet_info.model(aligned_cat.detach())


        # -- no error back to unet --
        # rec = rec.detach()

        # -- update unet --
        # if self.training: self.update_unet(aligned,residual)

        # -- update global step --
        if self.training:
            self.global_step += 1

        # -- return all components --
        return aligned,aligned_ave,denoised,denoised_ave,aligned_filters,denoised_filters

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

class BurstAlignSTN(nn.Module):

    def __init__(self, align_info, denoiser_info, use_alignment = True, skip_middle = True):
        super(BurstAlignSTN, self).__init__()
        self.align_info = align_info
        self.denoiser_info = denoiser_info
        self.skip_middle = skip_middle
        self.use_alignment=use_alignment
        self.global_step = 0
        # self.ot_loss = LossOT()

    def forward(self, burst):
        """
        :param burst: burst[:,1] ~ burst[:,N], shape: [N, batch, 3, height, width]
        """
        # -- init --
        CRITIC_UD = 1
        N,B,C,H,W = burst.shape
        stack = rearrange(burst,'n b c h w -> b n c h w')
        mid_img_r = burst[N//2].unsqueeze(1).repeat(1,N,1,1,1)
        kpn_cat = rearrange(burst,'n b c h w -> b (n c) h w')

        # -- align images --
        aligned = self.align_info.model(stack)
        aligned_ave = torch.mean(aligned,dim=1)
        aligned_filters = torch.zeros(B,N,1,1,1)

        # -- denoise --
        aligned_stack = aligned.detach()
        # both_stack = torch.cat([aligned,kpn_stack],dim=2)
        # both_stack = torch.cat([mid_img_r,kpn_stack],dim=2)
        # aligned_cat = rearrange(both_stack,'b n c h w -> b (n c) h w')
        aligned_cat = rearrange(aligned_stack,'b n c h w -> b (n c) h w')
        if self.use_alignment:
            denoised,denoised_ave,denoised_filters = self.denoiser_info.model(aligned_cat,aligned_stack)
        else:
            denoised,denoised_ave,denoised_filters = self.denoiser_info.model(kpn_cat,kpn_stack)

        # -- no error back to unet --
        # rec = rec.detach()

        # -- update unet --
        # if self.training: self.update_unet(aligned,residual)

        # -- update global step --
        if self.training:
            self.global_step += 1

        # -- return all components --
        return aligned,aligned_ave,denoised,denoised_ave,aligned_filters,denoised_filters

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


class BurstAlignN2N(nn.Module):

    def __init__(self, kpn, unet_info, noise_critic, skip_middle = True):
        super(BurstAlignN2N, self).__init__()
        self.kpn = kpn
        self.unet_info = unet_info
        self.skip_middle = skip_middle
        self.global_step = 0
        self.one = torch.FloatTensor([1.]).to(self.noise_critic.one.device)
        self.unet_start_iter = 15 * 1000
        self.unet_interval_iter = 1
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
        aligned,aligned_ave,temporal_loss,filters = self.kpn(kpn_cat,kpn_stack)

        # -- denoise --
        aligned_d = aligned.detach()
        denoised = torch.stack([self.unet_info.model(aligned_d[:,i]) for i in range(N)],dim=1)
        rec = torch.mean(denoised,dim=1)

        # -- no error back to unet --
        rec = rec.detach()

        # -- update unet --
        if self.update_unet_iter(): self.update_unet(aligned,residual)

        # -- update global step --
        if self.training:
            self.global_step += 1

        # -- return all components --
        return aligned,aligned_ave,denoised,rec,filters

    def update_unet_iter(self):
        update_bool = self.training
        update_bool = update_bool and (self.global_step > self.unet_start_iter)
        update_bool = update_bool and (self.global_step % self.unet_interval_iter) == 0
        return update_bool
        
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

class BurstRecLoss(nn.Module):
    """
    loss function of KPN
    """
    def __init__(self, coeff_ave=1.0, coeff_burst=100.0, coeff_ot=100.0, gradient_L1=True,
                 alpha=0.9998, beta=0.9772, reg=0.5, S=None):
        super(BurstRecLoss, self).__init__()
        self.coeff_ave = coeff_ave
        self.coeff_burst = coeff_burst
        self.coeff_ot = coeff_ot
        self.alpha,self.beta = alpha,beta

        use_tensor_grad = False # experiment using unsup, blind, no OT, rec_img for losses (no gt_img), noise critic, [with & without tensor_grad] show 23.46 v 25.71 PSNR
        self.loss_rec = LossRec(gradient_L1,use_tensor_grad)
        self.loss_burst = LossRecBurst(gradient_L1,use_tensor_grad)

    def forward(self, burst, burst_ave, ground_truth, global_step):
        """
        forward function of loss_func
        :param burst: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param burst_ave: \frac{1}{N} \sum_i^N frame_i, shape: [batch, 3, height, width]
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        # -- init --
        CRITIC_UD = 1
        B,N,C,H,W = burst.shape

        # -- alignment averages loss via MSE --
        loss_ave = self.loss_rec(burst_ave, ground_truth)
        loss_ave *= self.coeff_ave

        # -- alignment loss for each frame via MSE --
        loss_burst = self.loss_burst(burst, ground_truth)
        loss_burst *= self.coeff_burst

        return loss_ave, loss_burst

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
               

class EntropyLoss(nn.Module):
    """
    Compute the Entropy of the Module
    """

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        eps = 1e-15
        b = x * torch.log( x + eps )
        b = -1.0 * b.mean(dim=1)
        return b.mean()

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

    def __init__(self,reg=0.5,K=3,skip_middle=False):
        super(LossOT,self).__init__()
        self.reg = reg
        self.K = K
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
        return self.ot_frame_pairwise_xbatch_bp(diffs)
        
    def ot_frame_pairwise_bp(self,diffs):
        ot_loss = 0
        for b in range(BS):
            pairs,S = self._select_pairs(S,N)
            for (i,j) in pairs:
                di,dj = diffs[b,i],diffs[b,j]
                M = torch.sum(torch.pow(di.unsqueeze(1) - dj,2),dim=-1)
                ot_loss += sink_stabilized(M, self.reg)
        ot_loss /= S*BS
        return ot_loss
        
    def ot_frame_pairwise_xbatch_bp(self,residuals,reg=0.5,K=3):
        """
        :param residuals: shape [B N D C]
        """
        
        # -- init --
        B,N,D,C = residuals.shape
    
        # -- create triplets
        S = B*K
        indices,S = create_ot_indices(B,N,S)
    
        # -- compute losses --
        ot_loss = 0
        for (bi,bj,i,j) in indices:
            ri,rj = residuals[bi,i],residuals[bj,j]
            M = torch.sum(torch.pow(ri.unsqueeze(1) - rj,2),dim=-1)
            loss = sink_stabilized(M,reg)
            weight = ( torch.mean(ri) + torch.mean(rj) ) / 2
            ot_loss += loss * weight.item()
        return ot_loss / len(indices)
    
    def create_ot_indices(B,N,S):
        indices = []
        for i in range(N):
            for j in range(N):
                if i > j: continue
                for bi in range(B):
                    for bj in range(B):
                        if bi > bj: continue
                        index = (bi,bj,i,j)
                        indices.append(index)
    
        P = len(indices)
        indices = list(set(indices))
        assert P == len(indices), "We only created the list with unique elements"
        if S is None: S = P
        r_idx = npr.choice(range(P),S)
        s_indices = [indices[idx] for idx in r_idx]
        return s_indices,S
        
        

class NoiseCriticModel():

    def __init__(self,disc,optim,sim_params,device,p_lambda):
        self.disc = disc
        self.optim = optim
        self.sim_params = sim_params
        self.device = device
        self.p_lambda = p_lambda
        self.one = torch.FloatTensor([1.]).to(device)
        self.mone= -1 * self.one
        self.global_step = 0

    def compute_residual_loss(self,denoised,noisy_img,use_mean=True):
        """
        :params denoised: the (B,N,C,H,W) denoised images
        :params gt_img: the (B,C,H,W) noisy image to compare with
        """
        N = denoised.shape[1]
        residuals = denoised - noisy_img.unsqueeze(1).repeat(1,N,1,1,1)
        return self.compute_fake_loss(residuals,use_mean=use_mean)

    def compute_fake_loss(self,fake,use_mean=True):

        # -- freeze params --
        self.disc.zero_grad()
        for p in self.disc.parameters():
            p.requires_grad = False

        # -- reshape fake data --
        fake = rearrange(fake,'b n c h w -> (b n) c h w')

        # -- init --
        one = self.one

        # -- compute loss --
        output = self.disc(fake).view(-1)
        if use_mean: error_gen = output.mean(0).view(1)
        else: error_gen = output

        # -- un-freeze params --
        self.disc.zero_grad()
        for p in self.disc.parameters():
            p.requires_grad = True

        return -error_gen

    def calc_gradient_penalty(self, fake, real=None):

        # -- shape info --
        device = self.device
        B,C,H,W = fake.shape
        if real is None: real = self.sim_noise(fake.shape)

        # -- compute alpha for interpolation --
        alpha = torch.rand(B, 1)
        alpha = alpha.expand(B, int(real.nelement()/B) ).contiguous()
        alpha = alpha.view(B, C, H, W)
        alpha = alpha.to(device)

        # -- interpolate data --
        interpolates = alpha * real.detach() + ((1 - alpha) * fake.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # -- forward thru critic --
        disc_interpolates = self.disc(interpolates)

        # -- compute the gradients --
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)                              
        # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.p_lambda
        gradient_penalty = ((gradients.norm(2, dim=1)) ** 2).mean() * self.p_lambda

        # -- return norm --
        return gradient_penalty

    def update_disc(self,fake,real=None):
        """
        Maximize Discriminator
        """
        # -- init learn --
        self.disc.zero_grad()
        self.optim.zero_grad()

        # -- reshape fake data --
        fake = rearrange(fake,'b n c h w -> (b n) c h w')

        # -- init info --
        B,C,H,W = fake.shape
        one,mone = self.one,self.mone
        if real is None: real = self.sim_noise(fake.shape)
        
        # -- (i) real samples --
        output = self.disc(real).view(-1)
        err_disc_real = output.mean(0).view(1)
        D_x = output.mean().item()

        # -- (ii) fake samples --
        output = self.disc(fake.detach()).view(-1)
        err_disc_fake = output.mean(0).view(1)
        D_G_z1 = output.mean().item()
        
        # -- (iii) compute gradient penalty --
        gradient_penalty = self.calc_gradient_penalty(fake,real)

        # -- compute difference --
        error_disc = err_disc_real - err_disc_fake + gradient_penalty
        error_disc.backward(one)
        self.optim.step()

        # -- re-init learn --
        self.disc.zero_grad()
        self.optim.zero_grad()

        return error_disc.item()


    def update_disc_wgan(self,fake,real=None):
        """
        Maximize Discriminator
        """
        # -- init learn --
        self.disc.zero_grad()
        self.optim.zero_grad()

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

        # # -- print to stdout --
        # if self.global_step % 1 == 0:
        #     print(f"NoiseCritic: [{error_disc.item()}]")

        # -- simplify function (WGan) --
        for p in self.disc.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        # -- re-init learn --
        self.disc.zero_grad()
        self.optim.zero_grad()

        return error_disc.item()

    def sim_noise(self,shape):
        if self.sim_params.noise_type == "gaussian":
            return self.sim_gaussian(shape)
        else:
            raise NotImplemented(f"Uknown noise type {self.sim_params.noise_type}")

    def sim_gaussian(self,shape):
        samples = torch.normal(self.sim_params.mean,self.sim_params.std * torch.ones(shape))
        samples = samples.to(self.device)
        return samples


