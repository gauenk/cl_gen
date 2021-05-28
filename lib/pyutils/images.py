
# -- python imports --
import numpy as np
from PIL import Image

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms.functional as tvF

def print_tensor_stats(prefix,tensor):
    stats_fmt = (tensor.min().item(),tensor.max().item(),
                 tensor.mean().item(),tensor.std().item())
    stats_str = "%2.2e,%2.2e,%2.2e,%2.2e" % stats_fmt
    print(prefix,stats_str)

def np_log(np_array):
    if type(np_array) is not np.ndarray:
        if type(np_array) is not list:
            np_array = [np_array]
        np_array = np.array(np_array)
    return np.ma.log(np_array).filled(-np.infty)

def mse_to_psnr(mse):
    if isinstance(mse,float):
        psrn = 10 * np_log(1./mse)[0]/np_log(10)[0]
    else:
        psrn = 10 * np_log(1./mse)/np_log(10)
    return psrn

def rescale_noisy_image(img):
    img = img + 0.5
    return img

def add_noise(noise,pic):
    noisy_pic = pic + noise
    return noisy_pic

def adc_forward(cfg,image):
    params = cfg.noise_params['qis']
    pix_max = 2**params['nbits'] - 1
    image = torch.round(image)
    image = torch.clamp(image, 0, pix_max)
    return image

def normalize_image_to_zero_one(img):
    img = img.clone()
    img -= img.min()
    img /= img.max()
    return img

def images_to_psnrs(img1,img2):
    B = img1.shape[0]
    mse = F.mse_loss(img1.detach().cpu(),img2.detach().cpu(),reduction='none').reshape(B,-1)
    mse = torch.mean(mse,1).detach().numpy() + 1e-16
    psnrs = mse_to_psnr(mse)
    return psnrs

def save_image(images,fn,normalize=True,vrange=None):
    if len(images.shape) > 4:
        C,H,W = images.shape[-3:]
        images = images.reshape(-1,C,H,W)
    if vrange is None:
        tv_utils.save_image(images,fn,normalize=normalize)
    else:
        tv_utils.save_image(images,fn,normalize=normalize,range=vrange)

def read_image(image_path):
    image = Image.open(image_path)
    x = tvF.to_tensor(image)
    return x
