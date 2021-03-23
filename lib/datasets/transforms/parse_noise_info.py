
# -- torch imports --
from torchvision import transforms as tvT

# -- project imports --
from .misc import ScaleZeroMean
from .noise import AddGaussianNoiseSetN2N,GaussianBlur,AddGaussianNoise,AddPoissonNoiseBW,AddLowLightNoiseBW,AddHeteroGaussianNoise

__all__ = ['get_noise_transform']

def get_noise_transform(noise_info,noise_only=False,use_to_tensor=True):
    """
    The exemplar function for noise getting info
    """
    # -- get transforms --
    to_tensor = tvT.ToTensor()
    szm = ScaleZeroMean()
    noise = choose_noise_transform(noise_info)

    # -- create composition --
    comp = []
    if noise_only: comp = [noise]
    else:
        if use_to_tensor: comp = [to_tensor,noise,szm]
        else: comp = [noise,szm]
    transform = tvT.Compose(comp)

    return transform

def choose_noise_transform(noise_info):
    ntype = noise_info.ntype
    noise_params = noise_info[ntype]
    if ntype == "g":
        return get_g_noise(noise_params)
    if ntype == "hg":
        return get_hg_noise(noise_params)
    elif ntype == "ll":
        print("[parse_noise_info]: Check order of szm and noise fxn")
        return get_ll_noise(noise_params)
    elif ntype == "qis":
        print("[parse_noise_info]: Check order of szm and noise fxn")
        return get_qis_noise(noise_params)
    elif ntype == "msg":
        return get_msg_noise(noise_params)
    elif ntype == "msg_simcl":
        return get_msg_simcl_noise(noise_params)
    else:
        raise ValueError(f"Unknown noise_type [{ntype}]")

def get_g_noise(params):
    """
    Noise Type: Gaussian 
    """
    gaussian_noise = AddGaussianNoise(params['mean'],params['stddev'])
    return gaussian_noise

def get_hg_noise(params):
    """
    Noise Type: Heteroskedastic Gaussian N(x, \sigma_r + \sigma_s * x)
    """
    gaussian_noise = AddHeteroGaussianNoise(params['mean'],params['read'],params['shot'])
    return gaussian_noise
    
def get_ll_noise(params):
    """
    Noise Type: Low-Light  (LL)
    - Each N images is a low-light image with same alpha parameter
    """
    low_light_noise = LowLight(params['alpha'])
    return low_light_noise

def get_qis_noise(params):
    alpha,readout,nbits = params['alpha'],params['readout'],params['nbits']
    if readout > 1: readout /= 255. # rescale is necessary
    qis_noise = AddLowLightNoiseBW(alpha,readout,nbits)
    return qis_noise

def get_msg_noise(params):
    """
    Noise Type: Multi-scale Gaussian  (MSG)
    - Each N images has it's own noise level
    """
    std_range = (params['std_min'],params['std_max'])
    gaussian_n2n = AddGaussianNoiseSetN2N(params['N'],std_range)
    return gaussian_n2n

def get_msg_simcl_noise(params):
    """
    Noise Type: Multi-scale Gaussian  (MSG)
    - Each N images has it's own noise level

    plus contrastive learning augs
    - random crop (flip and resize)
    - color distortion
    - gaussian blur
    """
    N = params.N

    comp = []
    # -- random resize, crop, and flip --
    crop = th_transforms.RandomResizedCrop((self.size,self.size))
    comp += [crop]

    # -- flipping --
    # vflip = torchvision.transforms.RandomVerticalFlip(p=0.5)
    # comp += [vflip]
    hflip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
    comp += [hflip]

    # -- color jitter -- 
    s = params['s'] 
    c_jitter_kwargs = {'brightness':0.8*s,
                       'contrast':0.8*s,
                       'saturation':0.8*s,
                       'hue': 0.2*s}
    cjitter = torchvision.transforms.ColorJitter(**c_jitter_kwargs)
    cjitter = torchvision.transforms.RandomApply([cjitter], p=0.8)
    comp += [cjitter]

    # -- convert to gray --
    # gray = torchvision.transforms.RandomGrayscale(p=0.8)
    # comp += [gray]
    
    # -- gaussian blur --
    # gblur = GaussianBlur(size=3)
    # comp += [gblur]

    # -- convert to tensor --
    to_tensor = th_transforms.ToTensor()
    comp += [to_tensor]

    # -- center to zero mean, all within [-1,1] --
    # szm = ScaleZeroMean()
    # comp += [szm]

    # -- additive gaussian noise --
    # gaussian_n2n = AddGaussianNoiseRandStd(0,0,50)
    # comp += [gaussian_n2n]

    t = th_transforms.Compose(comp)

    # def t_n_raw(t,N,img):
    #     imgs = []
    #     for _ in range(N):
    #         imgs.append(t(img))
    #     imgs = torch.stack(imgs)
    #     return imgs
    # t_n = partial(t_n_raw,t,N)
    return t
