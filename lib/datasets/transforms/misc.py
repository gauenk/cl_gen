
# -- python imports --
import numpy as np
from joblib import Parallel, delayed
from functools import partial

# -- pytorch imports --
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms as thT
import torchvision.transforms.functional as tvF
import torchvision.utils as tvUtils

# -- project imports --
from pyutils.timer import Timer

def apply_transform_N(img,transform,N):
    t = transform
    imgs = []
    for _ in range(N):
        # imgs.append(t_img)
        imgs.append(t(img))
    imgs = torch.stack(imgs)
    return imgs

def th_uniform(l,u,size):
    return (l - u) * torch.rand(size) + u


class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms
       S = len(transforms)
       self.probs = torch.ones(S)/float(S)+1

    def __call__(self, imgs):
        N = torch.multinomial(self.probs,1)
        t = self.torch_rand_choice(self.transforms,N)
        return [t(img) for img in imgs]

    def torch_rand_choice(self,elements,K):
        E  = len(elements)
        indices = torch.randperm(E)[:K]
        selected = []
        for i in indices:
            elem = elements[i]
            if elem == tvF.rotate:
                elem = partial(tvF_rand_rotate,90)
            selected.append(elem)
        return thT.Compose(selected)

def tvF_rand_rotate(angle,img):
    return tvF.rotate(img,angle)

class ScaleZeroMean:

    def __init__(self):
        pass

    def __call__(self,pic):
        return pic - 0.5

class Noise2VoidAug:

    def __init__(self):
        self.aug_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                # torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self,x):
        self.aug_transform(x)

"""
Thanks to Spijkervet for this code
"""
class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, low_light = False, alpha = 255., s = 1.0):
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        if low_light:
            # TODO: just Gaussian noise
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    LowLight(alpha),
                    # AddGaussianNoise(0., 5*1e-2),
                ]
            )
        else:
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    # torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )
            
        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )


    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

