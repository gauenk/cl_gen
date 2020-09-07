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

# project imports
from settings import ROOT_PATH
from pyutils.misc import add_noise
from .transform import BlockGaussian,AddGaussianNoiseSet,ScaleZeroMean,AddGaussianNoiseSetN2N

def get_mnist_dataset(cfg,mode):
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
        data.val = DisentMNISTv1(root,N,noise_level,train=True)
        data.te = DisentMNISTv1(root,N,noise_level,train=False)
    else: raise ValueError(f"Unknown MNIST mode {mode}")
    loader = edict()
    loader_kwargs = {'batch_size': batch_size,
                     'shuffle':True,
                     'drop_last':True,
                     'num_workers':cfg[mode].workers,}
    loader.tr = DataLoader(data.tr,**loader_kwargs)
    loader.val = DataLoader(data.val,**loader_kwargs)
    loader.te = DataLoader(data.te,**loader_kwargs)
    return data,loader

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


