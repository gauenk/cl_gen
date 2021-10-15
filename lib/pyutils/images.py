
# -- python imports --
import numpy as np
from PIL import Image

# -- pytorch imports --
import torch
import torch.fft
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms.functional as tvF

def detached_cpu_tensor(img):
    if torch.is_tensor(img):
        return img.detach().cpu()
    else:
        return torch.Tensor(img)

def include_batch_dim(img):
    if img.ndim == 2:
        return img[None,None,:]
    elif img.ndim == 3:
        return img[None,:]
    elif img.ndim == 4: return img
    else:
        msg = f"[images.py:include_batch_dim] Uknown case images with ndim [{img.ndim}]"
        raise NotImplemented(msg)

def print_tensor_stats(prefix,tensor):
    # -- swap if necessary --
    if torch.is_tensor(prefix):
        tmp = prefix
        prefix = tensor
        tensor = prefix
    stats_fmt = (tensor.min().item(),tensor.max().item(),
                 tensor.mean().item(),tensor.std().item())
    stats_str = "[min,max,mean,std]: %2.2e,%2.2e,%2.2e,%2.2e" % stats_fmt
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
    img1 = detached_cpu_tensor(img1)
    img2 = detached_cpu_tensor(img2)

    assert img1.ndim == img2.ndim, "Equal number of dims"

    img1 = include_batch_dim(img1)
    img2 = include_batch_dim(img2)

    assert img1.ndim == 4, "Must be BatchSize x iDim0 x iDim1 x iDim2"

    B = img1.shape[0]
    mse = F.mse_loss(img1,img2,reduction='none').reshape(B,-1)
    mse = torch.mean(mse,1).detach().numpy() + 1e-16
    psnrs = mse_to_psnr(mse)
    return psnrs

def save_image(images,fn,normalize=True,vrange=None):
    if isinstance(images,str): # fix it: input are swapped of string and image
        tmp = images
        images = fn
        fn = tmp
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


def next_fast_len(n, factors=[2, 3, 5, 7]):
    '''
    Returns the minimum integer not smaller than n that can
    be written as a product (possibly with repettitions) of
    the given factors.
    '''
    best = float('inf')
    stack = [1]
    while len(stack):
        a = stack.pop()
        if a >= n:
            if a < best:
                best = a;
                if best == n:
                    break; # no reason to keep searching
        else:
            for p in factors:
                b = a * p;
                if b < best:
                    stack.append(b)
    return best

def torch_xcorr(signal_1, signal_2=None, factors=[2,3,5]):
    """
    signal: shape [B,D] for batch size B and field dimension D
    """
    # unpack shape
    B,D = signal_1.shape
    if signal_2 is None: signal_2 = signal_1

    # output target length of crosscorrelation
    x_cor_sig_length = D*2 - 1

    # get optimized array length for fft computation
    fast_length = next_fast_len(x_cor_sig_length, factors)

    # the last signal_ndim axes (1,2 or 3) will be transformed
    fft_1 = torch.fft.rfft(signal_1, fast_length, dim=-1)
    fft_2 = torch.fft.rfft(signal_2, fast_length, dim=-1)

    # take the complex conjugate of one of the spectrums.
    # Which one you choose depends on domain specific conventions
    fft_multiplied = torch.conj(fft_1) * fft_2

    # back to time domain. 
    prelim_correlation = torch.fft.irfft(fft_multiplied, dim=-1)

    # shift the signal to make it look like a proper crosscorrelation,
    # and transform the output to be purely real
    final_result = torch.roll(prelim_correlation, (fast_length//2,))

    # _,corr_ncc = normalized_cross_correlation(signal_1, signal_1,
    # True, 'mean', eps=1e-8)
    # corr_ncc = corr_ncc.numpy()
    # print(corr_ncc.shape)
    # corr_ncc = corr_ncc[0,:]
    # print(corr_ncc.shape)
    
    # corr_f = final_result[0,:].cpu().numpy()[:-1]
    # x = signal_1.cpu().numpy()[0,:]
    # print(x.shape)
    # corr_x = np.correlate(x,x,mode='same')
    # length = len(x)-1
    # acf_x = np.array([np.corrcoef(x[:-i],x[i:])[0,1] for i in range(1,length)])
    # print(corr_ncc)
    # print(corr_x)
    # print(corr_f)
    # print(corr_x.shape,corr_f.shape,acf_x.shape)
    # print(np.sum(corr_x),np.sum(corr_f))
    # print(acf_x)
    # print(corr_x/corr_f)
    # print("diff",np.sum(np.abs(corr_x - corr_f)))


    return final_result, torch.sum(final_result,dim=1)


def normalized_cross_correlation(x, y, return_map, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    if not return_map:
        return ncc

    return ncc, ncc_map
