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

# pytorch imports
import torch,torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms as th_transforms

# project imports
from settings import ROOT_PATH
from pyutils.misc import add_noise
from .transform import TransformsSimCLR

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

#
# Loading the datasets in a project
#

def get_cifar10_dataset(cfg,mode):
    root = cfg[mode].dataset.root
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
    else: raise ValueError(f"Unknown CIFAR10 mode {mode}")

    loader = edict()
    loader_kwargs = {'batch_size': batch_size,
                     'shuffle':True,
                     'drop_last':True,
                     'num_workers':cfg[mode].workers,}
    loader.tr = DataLoader(data.tr,**loader_kwargs)
    loader.val = DataLoader(data.val,**loader_kwargs)
    loader.te = DataLoader(data.te,**loader_kwargs)

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


