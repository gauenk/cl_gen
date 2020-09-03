"""
Contrastive Learning Generation
Kent Gauen

Imagenet Dataset Objects

"""

# python imports
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from pathlib import Path

# project imports
from settings import ROOT_PATH
from .transform import AddGaussianNoiseSet,ScaleZeroMean


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

    def __init__(self, root: str, split: str = 'train', transform=None, target_transform=None):
        root = Path(root) / Path("imagenet")
        transform = th_transforms.Compose([torchvision.transforms.Resize(size=256),
                                           th_transforms.ToTensor(),
                                           ScaleZeroMean(),
                                           AddGaussianNoiseSet(N,std=noise_level),
                                           ])
        th_trans = th_transforms.Compose([torchvision.transforms.Resize(size=256),
                                          ScaleZeroMean(),
                                           th_transforms.ToTensor()
                                           ])
        super(DisentImageNetv1, self).__init__( root, split, None, transform=transform,
                                                target_transform=target_transform)
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

