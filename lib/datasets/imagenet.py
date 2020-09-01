"""
Contrastive Learning Generation
Kent Gauen

Imagenet Dataset Objects

"""

# python imports
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet

# project imports
from settings import ROOT_PATH

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
