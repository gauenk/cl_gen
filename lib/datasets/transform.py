"""
Thanks to Spijkervet for this code
"""

import numpy as np
import numpy.random as npr
import torch
import torchvision


class LowLight:

    def __init__(self,alpha,seed=None):
        self.alpha = alpha
        self.seed = seed

    def __call__(self,pic):
        low_light_pic = torch.poisson(self.alpha*pic,generator=self.seed)/self.alpha
        return low_light_pic

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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=10e-3):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianNoiseSet(object):
    def __init__(self, N, mean=0., std=1e-2):
        self.N = N
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        pics = []
        # print(tensor.shape)
        for n in range(self.N):
            pic = tensor + torch.randn(tensor.size()) * self.std + self.mean
            pics.append(pic)
        return pics
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class BlockGaussian:

    def __init__(self,N,mean=0.,std=1e-1, mean_size=.4):
        self.N = N # number of transforms
        self.mean = mean
        self.std = std
        self.mean_size = mean_size

    def __call__(self, pic):
        (h,w) = pic.shape
        b_h = int(self.mean_size * h)
        b_w = int(self.mean_size * w)
        pics = []
        for n in range(self.N):
            pic_n = pic.clone()
            y = npr.randint(h)
            x = npr.randint(w)
            y1 = np.clip(y - b_h // 2, 0, h)
            y2 = np.clip(y + b_h // 2, 0, h)
            x1 = np.clip(x - b_w // 2, 0, w)
            x2 = np.clip(x + b_w // 2, 0, w)
            mask = npr.normal(self.mean,self.std,(y2-y1,x2-x1))
            mask = torch.Tensor(mask).to(pic.device)
            mask = torch.clamp(mask,0,1.)
            pic_n[y1:y2, x1:x2] = mask
            pics.append(pic_n)
        return pics
