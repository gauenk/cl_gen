"""
Modified CelebA datasets

"""

# python imports
from PIL import Image
from functools import partial
from easydict import EasyDict as edict
import numpy.random as npr

# pytorch imports
import torch,torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision import transforms as th_transforms

# project imports
from settings import ROOT_PATH
from pyutils.misc import add_noise
from datasets.transforms import BlockGaussian,AddGaussianNoiseSet

def get_celeba_dataset(cfg,mode):
    root = cfg[mode].dataset.root
    data = edict()
    if mode == 'cls':
        batch_size = cfg.cls.batch_size
        transform = th_transforms.Compose([th_transforms.ToTensor()])
        data.tr = CelebA(root,split='train',transform=transform)
        data.val = CelebA(root,split='valid',transform=transform)
        data.te = CelebA(root,split='test',transform=transform)
        data.all = CelebA(root,split='all',transform=transform)
    elif mode == "disent":
        batch_size = cfg.disent.batch_size
        N = cfg.disent.N
        data.tr = DisentCelebAv1(root,N,split='train')
        data.val = DisentCelebAv1(root,N,split='valid')
        data.te = DisentCelebAv1(root,N,split='test')
        data.all = DisentCelebAv1(root,N,split='all')
    else: raise ValueError(f"Unknown CelebA mode {mode}")
    loader = edict()
    loader_kwargs = {'batch_size': batch_size,
                     'shuffle':True,'drop_last':True,
                     'num_workers':cfg[mode].workers,}
    loader.tr = DataLoader(data.tr,**loader_kwargs)
    loader.val = DataLoader(data.val,**loader_kwargs)
    loader.te = DataLoader(data.te,**loader_kwargs)
    return data,loader

class DisentCelebAv1(CelebA):
    """
    This wrapper just rewrites the original Imagenet class 
    to produce a pair of transformed observations rather than just one.
    We overwrite:
    __getitem__
    """



    def __init__(self, root, N, split='train', low_light=False,
                 transform=None, target_transform=None,download=False):
        # transform = BlockGaussian(N)
        transform = AddGaussianNoiseSet(N)
        self.__class__.__name__ = "celeba"
        super(DisentCelebAv1, self).__init__( root, split=split, transform=transform,
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

        th_img = img.float().div(255)
        img_set = self.transform(th_img)

        if self.target_transform is not None:
            target = self.target_transform(target,index)

        return img_set, th_img


