"""
Contrastive Learning Generation
Kent Gauen

Imagenet Dataset Objects

"""

# python imports
from PIL import Image
from functools import partial
from easydict import EasyDict as edict
import numpy.random as npr
from pathlib import Path
import numpy as np

# pytorch imports
import torch,torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms as th_transforms
from torch.utils.data.distributed import DistributedSampler


# project imports
from settings import ROOT_PATH
from pyutils.misc import add_noise
from .transform import TransformsSimCLR,AddGaussianNoiseSet,ScaleZeroMean,AddGaussianNoiseSetN2N,GaussianBlur

class ClCIFAR10(CIFAR10):
    """
    This wrapper just rewrites the original Imagenet class 
    to produce a pair of transformed observations rather than just one.
    We overwrite:
    __getitem__
    """

    def __init__(self, root, image_size, train=True, low_light=False, transform=None, target_transform=None,
                 download=False):
        
        transform = TransformsSimCLR(size=image_size,low_light=low_light)
        super(ClCIFAR10, self).__init__( root, train=train, transform=transform,
                                          target_transform=target_transform,
                                          download=download)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        th_img = torch.from_numpy(img).float().div(255).permute((2, 0, 1)).contiguous()
        img = Image.fromarray(img)
        img_i,img_j = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target,index)

        return (img_i, img_j), th_img


class ImgRecCIFAR10(CIFAR10):
    """
    This wrapper just rewrites the original Imagenet class 
    to produce a pair of transformed observations rather than just one.
    We overwrite:
    __getitem__
    """

    def __init__(self, root, transforms, train=True):
        super(ImgRecCIFAR10, self).__init__( root, train=train, transform=None,
                                         download=False)
        self.pic_transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = Image.fromarray(self.data[index]), int(self.targets[index])
        transforms = self.pic_transforms[index]
        size = (33,33)
        to_tensor = th_transforms.Compose([th_transforms.Grayscale(),
                                           torchvision.transforms.Resize(size=size),
                                           th_transforms.ToTensor()])
        img = to_tensor(img)

        img_trans = []
        for transform_i in transforms:
            img_i = add_noise(transform_i,img)
            img_trans.append(img_i)
        return img_trans,index


class DisentCIFAR10v1(CIFAR10):
    """
    This wrapper just rewrites the original Imagenet class 
    to produce a pair of transformed observations rather than just one.
    We overwrite:
    __getitem__
    """



    def __init__(self, root, N, noise_level=1e-2, train=True, low_light=False,
                 transform=None, target_transform=None,download=False):
        # transform = BlockGaussian(N)
        transform = th_transforms.Compose([torchvision.transforms.Resize(size=32),
                                           th_transforms.ToTensor(),
                                           ScaleZeroMean(),
                                           AddGaussianNoiseSetN2N(N,(0,50.))
                                           ])
        val_trans = th_transforms.Compose([torchvision.transforms.Resize(size=32),
                                           th_transforms.ToTensor(),
                                           ScaleZeroMean(),
                                           AddGaussianNoiseSetN2N(N,(24.99,25.01))
                                           ])
        th_trans = th_transforms.Compose([torchvision.transforms.Resize(size=32),
                                          th_transforms.ToTensor(),
                                          ScaleZeroMean(),
                                           ])

        self.__class__.__name__ = "cifar10"
        super(DisentCIFAR10v1, self).__init__( root, train=train, transform=transform,
                                          target_transform=target_transform,
                                          download=download)
        self.val_trans = val_trans
        self.th_trans = th_trans

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        pil_pic = Image.fromarray(img)
        img_set = self.transform(pil_pic)
        th_img = self.th_trans(pil_pic)

        if self.target_transform is not None:
            target = self.target_transform(target,index)

        return img_set, th_img

class DenoiseCIFAR10(CIFAR10):

    def __init__(self,root,N,noise_type,noise_params,train=True):
        self.N = N
        self.noise_type = noise_type
        self.noise_params = noise_params
        trans = self._get_noise_transform(noise_type,noise_params,N)
        th_trans = self._get_th_img_trans()
        self.__class__.__name__ = "cifar10"
        super(DenoiseCIFAR10, self).__init__( root, train=train,
                                              transform=trans)
        self.th_trans = th_trans

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = Image.fromarray(self.data[index]), int(self.targets[index])
        img_set = self.transform(img)
        th_img = self.th_trans(img)

        if self.target_transform is not None:
            target = self.target_transform(target,index)

        return img_set, th_img

    def _get_noise_transform(self,noise_type,params,N):
        if noise_type == "g":
            return self._get_g_noise(params,N)
        elif noise_type == "ll":
            return self._get_ll_noise(params,N)
        elif noise_type == "msg":
            print("Loading msg transforms")
            return self._get_msg_noise(params,N)
        elif noise_type == "msg_simcl":
            print("Loading msg_simcl transforms")
            return self._get_msg_simcl_noise(params,N)
        else:
            raise ValueError(f"Unknown noise_type [{noise_type}]")

    def _get_g_noise(self,params,N):
        """
        Noise Type: Gaussian  (LL)
        - Each N images has Gaussian noise from with same parameters
        """
        resize = torchvision.transforms.Resize(size=32)
        to_tensor = th_transforms.ToTensor()
        szm = ScaleZeroMean()
        gaussian_noise = AddGaussianNoiseSet(N,params['mean'],params['stddev'])
        comp = [resize,to_tensor,szm,gaussian_noise]
        t = th_transforms.Compose(comp)
        return t

    def _get_ll_noise(self,params,N):
        """
        Noise Type: Low-Light  (LL)
        - Each N images is a low-light image with same alpha parameter
        """
        raise NotImplemented()

    def _get_msg_noise(self,params,N):
        """
        Noise Type: Multi-scale Gaussian  (MSG)
        - Each N images has it's own noise level
        """
        resize = torchvision.transforms.Resize(size=32)
        to_tensor = th_transforms.ToTensor()
        szm = ScaleZeroMean()
        gaussian_n2n = AddGaussianNoiseSetN2N(N,(0,50.))
        comp = [resize,to_tensor,szm,gaussian_n2n]
        t = th_transforms.Compose(comp)
        return t

    def _get_msg_simcl_noise(self,params,N):
        """
        Noise Type: Multi-scale Gaussian  (MSG)
        - Each N images has it's own noise level

        plus contrastive learning augs
        - random crop (flip and resize)
        - color distortion
        - gaussian blur
        """
        comp = []

        # -- random resize, crop, and flip --
        crop = torchvision.transforms.RandomResizedCrop(size=32)
        comp += [crop]

        # -- flipping --
        vflip = torchvision.transforms.RandomVerticalFlip(p=0.5)
        comp += [vflip]
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
        gray = torchvision.transforms.RandomGrayscale(p=0.8)
        comp += [gray]
        
        # -- gaussian blur --
        gblur = GaussianBlur(size=3)

        # -- convert to tensor --
        to_tensor = th_transforms.ToTensor()
        comp += [to_tensor]

        # -- center to zero mean, all within [-1,1] --
        szm = ScaleZeroMean()
        comp += [szm]

        # -- additive gaussian noise --
        gaussian_n2n = AddGaussianNoiseSetN2N(N,(0,50.))
        comp += [gaussian_n2n]

        t = th_transforms.Compose(comp)
        return t
        
    def _get_th_img_trans(self):
        t = th_transforms.Compose([torchvision.transforms.Resize(size=32),
                                   th_transforms.ToTensor()
        ])
        return t


    

#
# Loading the datasets in a project
#

def get_cifar10_dataset(cfg,mode):
    if mode == "denoising" or mode == "simcl" or mode == "simcl_cls":
        root = cfg.dataset.root
    else:
        root = cfg[mode].dataset.root
    root = Path(root)/Path("cifar10")
    data = edict()
    if mode == 'cl':
        batch_size = cfg.cl.batch_size
        low_light = cfg.cl.dataset.transforms.low_light
        data.tr = ClCIFAR10(root,cfg.cl.image_size,train=True,low_light=low_light)
        data.val = ClCIFAR10(root,cfg.cl.image_size,train=True,low_light=low_light)
        data.te = ClCIFAR10(root,cfg.cl.image_size,train=False,low_light=low_light)
    elif mode == 'cls':
        download = cfg.cls.dataset.download
        batch_size = cfg.cls.batch_size
        transform = th_transforms.Compose([th_transforms.ToTensor()])
        data.tr = CIFAR10(root,train=True,transform=transform,download=download)
        data.val = CIFAR10(root,train=True,transform=transform,download=download)
        data.te = CIFAR10(root,train=False,transform=transform,download=download)
    elif mode == "imgrec":
        batch_size = cfg.imgrec.batch_size
        cifar_transforms = get_cifar10_transforms(cfg)
        data.tr = ImgRecCIFAR10(root,cifar_transforms.tr,train=True)
        data.val = ImgRecCIFAR10(root,cifar_transforms.tr,train=True)
        data.te = ImgRecCIFAR10(root,cifar_transforms.te,train=False)
        data.transforms = cifar_transforms
        batch_size = cfg.disent.batch_size
    elif mode == "disent":
        batch_size = cfg.disent.batch_size
        N = cfg.disent.N
        noise_level = cfg.disent.noise_level
        data.tr = DisentCIFAR10v1(root,N,noise_level,train=True)
        data.val = DisentCIFAR10v1(root,N,noise_level,train=True)
        data.te = DisentCIFAR10v1(root,N,noise_level,train=False)
    elif mode == "simcl":
        batch_size = cfg.batch_size
        N = cfg.N
        noise_type = cfg.noise_type
        noise_params = cfg.noise_params[noise_type]
        data.tr = DenoiseCIFAR10(root,N,noise_type,noise_params,train=True)
        data.val = DenoiseCIFAR10(root,N,noise_type,noise_params,train=True)
        data.val.data = data.val.data[0:2000]
        data.val.targets = data.val.targets[0:2000]
        data.te = DenoiseCIFAR10(root,N,noise_type,noise_params,train=False)
    else: raise ValueError(f"Unknown CIFAR10 mode {mode}")

    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader

def get_cifar10_transforms(cfg):
    cls_batch_size = cfg.cls.batch_size
    cfg.cls.batch_size = 1
    data,loader = get_cifar10_dataset(cfg,'cls')
    cfg.cls.batch_size = cls_batch_size

    transforms = edict()
    transforms.tr = get_dataset_transforms(cfg,data.tr.data)
    transforms.te = get_dataset_transforms(cfg,data.te.data)

    return transforms

def get_dataset_transforms(cfg,data):
    import numpy as np
    noise_levels = cfg.imgrec.dataset.noise_levels
    noise_data = []
    for noise_level in noise_levels:
        shape = (len(data),3,33,33)
        means = torch.zeros(shape)
        noise_i = torch.normal(means,noise_level)
        noise_data.append(noise_i)
    noise_data = torch.stack(noise_data,dim=1)
    return noise_data

    # trans = []
    # for m in range(len(data)):
    #     trans_m = []
    #     for n in range(len(noise_levels)):
    #         noise = noise_data[n][m].numpy()
    #         trans_mn = partial(add_noise,noise)
    #         trans_m.append(trans_mn)
    #     trans.append(trans_m)
    # return trans


def get_loader(cfg,data,batch_size,mode):
    if cfg.use_ddp:
        loader = get_loader_ddp(cfg,data)
    else:
        loader = get_loader_serial(cfg,data,batch_size,mode)
    return loader
    
def collate_fn(batch):
    noisy,clean = zip(*batch)
    noisy = torch.stack(noisy,dim=1)
    clean = torch.stack(clean,dim=0)
    return noisy,clean

def get_loader_serial(cfg,data,batch_size,mode):

    loader_kwargs = {'batch_size': batch_size,
                     'shuffle':True,
                     'drop_last':True,
                     'num_workers':cfg.num_workers,
                     'pin_memory':True}
    if cfg.use_collate:
        loader_kwargs['collate_fn'] = collate_fn

    loader = edict()
    loader.tr = DataLoader(data.tr,**loader_kwargs)
    loader_kwargs['drop_last'] = False
    loader.val = DataLoader(data.val,**loader_kwargs)
    loader.te = DataLoader(data.te,**loader_kwargs)
    return loader

def get_loader_ddp(cfg,data):
    loader = edict()
    loader_kwargs = {'batch_size':cfg.batch_size,
                     'shuffle':False,
                     'drop_last':True,
                     'num_workers': 1, #cfg.num_workers,
                     'pin_memory':True}
    if cfg.use_collate:
        loader_kwargs['collate_fn'] = collate_fn

    ws = cfg.world_size
    r = cfg.rank
    loader = edict()

    sampler = DistributedSampler(data.tr,num_replicas=ws,rank=r)
    loader_kwargs['sampler'] = sampler
    loader.tr = DataLoader(data.tr,**loader_kwargs)

    sampler = DistributedSampler(data.val,num_replicas=ws,rank=r)
    loader_kwargs['sampler'] = sampler
    loader.val = DataLoader(data.val,**loader_kwargs)

    sampler = DistributedSampler(data.te,num_replicas=ws,rank=r)
    loader_kwargs['drop_last'] = False
    loader_kwargs['sampler'] = sampler
    loader.te = DataLoader(data.te,**loader_kwargs)

    return loader

