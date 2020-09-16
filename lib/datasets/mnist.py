"""
Modified MNIST datasets

"""

# python imports
from PIL import Image
from functools import partial
from easydict import EasyDict as edict
import numpy.random as npr

# pytorch imports
import torch,torchvision
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms as th_transforms
from torch.utils.data.distributed import DistributedSampler

# project imports
from settings import ROOT_PATH
from pyutils.misc import add_noise
from .transform import BlockGaussian,AddGaussianNoiseSet,ScaleZeroMean,AddGaussianNoiseSetN2N

def get_mnist_dataset(cfg,mode):
    if mode == "denoising":
        root = cfg.dataset.root
    else:
        root = cfg[mode].dataset.root
    data = edict()
    if mode == 'cls':
        batch_size = cfg.cls.batch_size
        download = cfg.cls.dataset.download
        transform = th_transforms.Compose([th_transforms.ToTensor()])
        data.tr = MNIST(root,train=True,transform=transform,download=download)
        data.val = MNIST(root,train=True,transform=transform,download=download)
        data.te = MNIST(root,train=False,transform=transform,download=download)
    elif mode == "disent":
        batch_size = cfg.disent.batch_size
        N = cfg.disent.N
        noise_level = cfg.disent.noise_level
        data.tr = DisentMNISTv1(root,N,noise_level,train=True)
        data.val = DisentMNISTv1(root,N,noise_level,train=False)
        # subset to only 1000
        data.val.data = data.val.data[0:1000]
        data.val.targets = data.val.targets[0:1000]
        data.te = DisentMNISTv1(root,N,noise_level,train=False)
    elif mode == "denoising":
        batch_size = cfg.batch_size
        N = cfg.N
        noise_type = cfg.noise_type
        noise_params = cfg.noise_params[noise_type]
        data.tr = DenoiseMNIST(root,N,noise_type,noise_params,train=True)
        data.val = DenoiseMNIST(root,N,noise_type,noise_params,train=False)
        data.val.data = data.val.data[0:1000]
        data.val.targets = data.val.targets[0:1000]
        data.te = DenoiseMNIST(root,N,noise_type,noise_params,train=False)
    else: raise ValueError(f"Unknown MNIST mode {mode}")

    loader = get_loader(cfg,data,batch_size,mode)

    return data,loader


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
    if mode == "disent" or mode == "denoising":
        loader_kwargs['collate_fn'] = collate_fn

    loader = edict()
    loader.tr = DataLoader(data.tr,**loader_kwargs)
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
    loader_kwargs['sampler'] = sampler
    loader.te = DataLoader(data.te,**loader_kwargs)

    return loader


# class DenoisingSetCollate():

#     def __init__(self,data):
        

class DisentMNISTv1(MNIST):
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
        th_trans = th_transforms.Compose([torchvision.transforms.Resize(size=32),
                                           th_transforms.ToTensor()
                                           ])
        self.__class__.__name__ = "mnist"
        super(DisentMNISTv1, self).__init__( root, train=train, transform=transform,
                                             target_transform=target_transform,
                                             download=download)
        self.th_trans = th_trans

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        pil_pic = Image.fromarray(img.to('cpu').detach().numpy())
        img_set = self.transform(pil_pic)
        th_img = self.th_trans(pil_pic)

        if self.target_transform is not None:
            target = self.target_transform(target,index)

        return img_set, th_img


class DenoiseMNIST(MNIST):
    """
    This wrapper just rewrites the original Imagenet class 
    to produce a pair of transformed observations rather than just one.
    We overwrite:
    __getitem__
    """



    def __init__(self, root, N, noise_type, noise_params, train=True,
                 transform=None, target_transform=None,download=False):
        transform = self._get_noise_transform(noise_type,noise_params,N)
        th_trans = self._get_th_img_trans()
        self.__class__.__name__ = "mnist"
        super(DenoiseMNIST, self).__init__( root, train=train, transform=transform,
                                             target_transform=target_transform,
                                             download=download)
        self.th_trans = th_trans

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        pil_pic = Image.fromarray(img.to('cpu').detach().numpy())
        img_set = self.transform(pil_pic)
        th_img = self.th_trans(pil_pic)

        if self.target_transform is not None:
            target = self.target_transform(target,index)

        return img_set, th_img

    def _get_noise_transform(self,noise_type,params,N):
        if noise_type == "g":
            return self._get_g_noise(params,N)
        elif noise_type == "ll":
            return self._get_ll_noise(params,N)
        elif noise_type == "msg":
            return self._get_msg_noise(params,N)
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

        
    def _get_th_img_trans(self):
        t = th_transforms.Compose([torchvision.transforms.Resize(size=32),
                                   th_transforms.ToTensor()
        ])
        return t
