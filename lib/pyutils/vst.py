
# -- python imports --
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange, repeat, reduce

# -- pytorch imports --
import torch
import torch.nn.functional as F

# -- [local] project imports --
from .misc import adc_forward,mse_to_psnr

def anscombe_forward(sample):
    return 2*torch.pow(sample+3./8,0.5)

def anscombe_backward(sample):
    inv = 0.25 * torch.pow(sample,2)
    inv -= 1./8
    inv += 0.25 * np.sqrt(3./2) * torch.pow(sample,-1)
    inv -= 11./8 * torch.pow(sample,-2)
    inv += 5./8 * np.sqrt(3./2) * torch.pow(sample,-3)
    return inv

anscombe = edict({'forward':anscombe_forward,
                  'backward':anscombe_backward})

def anscombe_nmlz_forward(cfg,sample):
    assert sample.min() >= 0., "Non-negative values only"
    alpha = cfg.noise_params['qis']['alpha']
    nmlz_sample = alpha*(sample)
    anscombe_sample = anscombe.forward(nmlz_sample)
    return anscombe_sample/alpha
    
def anscombe_nmlz_backward(cfg,anscombe_sample):
    assert anscombe_sample.min() >= 0., "Non-negative values only"
    alpha = cfg.noise_params['qis']['alpha']
    print(anscombe_sample)
    sample = anscombe.backward(anscombe_sample*alpha)
    print('sv',sample)
    print("s",sample.min().item(),sample.max().item(),sample.mean().item())
    adc_sample = adc_forward(cfg,sample)
    nmlz_sample = adc_sample / alpha
    return nmlz_sample

anscombe_nmlz = edict({'forward':anscombe_nmlz_forward,
                       'backward':anscombe_nmlz_backward})


def anscombe_test(cfg,burst):
    B = burst.shape[1]
    alpha = cfg.noise_params['qis']['alpha']

    # -- pre-processing --
    input_burst = alpha*(burst+0.5)

    # -- anscombe --
    a_burst = anscombe.forward(input_burst)
    a_burst = anscombe.backward(a_burst)

    # -- binarize --
    a_burst = adc_forward(cfg,a_burst)

    # -- post-processing --
    output_burst = a_burst / alpha - 0.5

    a_mse = F.mse_loss(output_burst,burst,reduction='none')
    a_mse = rearrange(a_mse,'n b c h w -> b (n c h w)')
    a_mse = torch.mean(a_mse,1).detach().cpu().numpy()
    a_mse_ave = np.mean(a_mse)
    a_mse_std = np.std(a_mse)
    a_psnr_ave = np.mean(mse_to_psnr(a_mse))
    a_psnr_std = np.std(mse_to_psnr(a_mse))
    print("MSE: %2.2f +/- %2.2f" %  ( a_mse_ave, a_mse_std ) )
    print("Anscombe PSNR: %2.2f +/- %2.2f" %  ( a_psnr_ave, a_psnr_std ) )

anscombe.test = anscombe_test
