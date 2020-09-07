"""
Contrastive Learning Generation
Kent Gauen

Imagenet Dataset Objects

"""

# python imports
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict

# pytorch imports
from torchvision.datasets import ImageNet
import torch,torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as th_transforms

# project imports
from settings import ROOT_PATH
from .transform import AddGaussianNoiseSet,ScaleZeroMean,AddGaussianNoiseSetN2N


class ClImageNet(ImageNet):
    """
    This wrapper just rewrites the original Imagenet class 
    to produce a pair of transformed observations rather than just one.
    We overwrite:
    __getitem__
    """

    def __init__(self, root: str, split: str = 'train', transform=None, target_transform=None):
        print("ROOT:", root)
        super(ClImageNet, self).__init__( root, split, None, transform=transform,
                                          target_transform=target_transform)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img)
        img_a = self.transform(img,'rand')
        img_b = self.transform(img,'rand')

        if self.target_transform is not None:
            target = self.target_transform(target,index)

        return img_a, img_b

class DisentImageNetv1(ImageNet):
    """
    This wrapper just rewrites the original Imagenet class 
    to produce a pair of transformed observations rather than just one.
    We overwrite:
    __getitem__
    """

    def __init__(self, root: str, N: int, noise_level: float, random_crop: bool, split: str = 'train', transform=None, target_transform=None):
        root = Path(root) / Path("imagenet/")
        if random_crop:
            transform = th_transforms.Compose([torchvision.transforms.Resize(size=256),
                                               torchvision.transforms.RandomCrop(size=256),
                                               th_transforms.ToTensor(),
                                               ScaleZeroMean(),
                                               AddGaussianNoiseSetN2N(N,(0,50.))
            ])
        else:
            transform = th_transforms.Compose([torchvision.transforms.Resize(size=256),
                                               th_transforms.ToTensor(),
                                               ScaleZeroMean(),
                                               AddGaussianNoiseSetN2N(N,(0,50.))
            ])
        th_trans = th_transforms.Compose([torchvision.transforms.Resize(size=256),
                                          th_transforms.ToTensor(),
                                          ScaleZeroMean(),
        ])
        super(DisentImageNetv1, self).__init__( root, split, None, transform=transform,
                                                target_transform=target_transform)
        self.th_trans = th_trans

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.imgs[index]
        pil_pic = Image.open(img)
        img_set = self.transform(pil_pic)
        th_img = self.th_trans(pil_pic)

        if self.target_transform is not None:
            target = self.target_transform(target,index)

        return img_set, th_img




def get_imagenet_dataset(cfg,mode):
    root = cfg[mode].dataset.root
    data = edict()
    if mode == "disent":
        batch_size = cfg.disent.batch_size
        N = cfg.disent.N
        noise_level = cfg.disent.noise_level
        random_crop = cfg.disent.random_crop
        # data.tr = DisentImageNetv1(root,N,noise_level,split="train")
        data.val = DisentImageNetv1(root,N,noise_level,random_crop,split="val")
        # data.te = DisentImageNetv1(root,N,noise_level,split="test")
    else: raise ValueError(f"Unknown ImageNet mode {mode}")

    loader = edict()
    loader_kwargs = {'batch_size': batch_size,
                     'shuffle':True,
                     'drop_last':True,
                     'num_workers':cfg[mode].workers,}
    # loader.tr = DataLoader(data.tr,**loader_kwargs)
    loader.val = DataLoader(data.val,**loader_kwargs)
    # loader.te = DataLoader(data.te,**loader_kwargs)

    return data,loader

