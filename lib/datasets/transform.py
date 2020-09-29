"""
Thanks to Spijkervet for this code
"""

import torch
import torchvision
import torch.nn.functional as F

def th_uniform(l,u,size):
    return (l - u) * torch.rand(size) + u

class LowLight:

    def __init__(self,alpha,seed=None):
        self.alpha = alpha
        self.seed = seed

    def __call__(self,pic):
        low_light_pic = torch.poisson(self.alpha*pic,generator=self.seed)/self.alpha
        return low_light_pic

class ScaleZeroMean:

    def __init__(self):
        pass

    def __call__(self,pic):
        return pic - 0.5


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
    def __init__(self, mean=0., std=1e-2):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        pic = torch.normal(tensor.add(self.mean),self.std)
        return pic
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianNoiseSet(object):
    def __init__(self, N, mean= 0., std=50):
        self.N = N
        self.mean = mean
        self.std = std / 255.
        
    def __call__(self, tensor):
        pics = tensor.add(self.mean)
        pics = torch.stack(self.N*[pics])
        return torch.normal(pics,self.std)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddGaussianNoiseSetN2N(object):

    def __init__(self, N, train_stddev_rng_range=(0,50.)):
        self.N = N
        self.train_stddev_rng_range = train_stddev_rng_range
        
    def __call__(self, tensor):

        shape = (self.N,) + tensor.shape

        # get noise term
        (minv,maxv) = self.train_stddev_rng_range
        rnoises = th_uniform(minv/255,maxv/255.,self.N).to(tensor.device)
        for i in range(len(shape)-1): rnoises = rnoises.unsqueeze(1)
        rnoises = rnoises.expand(shape)

        # get means
        means = tensor.expand(shape)

        # simulate noise
        pics = torch.normal(means,rnoises)

        return pics
    
    def __repr__(self):
        rng = self.train_stddev_rng_range
        return self.__class__.__name__ + '({})'.format(rng)

class GaussianBlur:

    def __init__(self,size,min_sigma=0.1,max_sigma=2.0,dim=2,channels=3):
        self.size = size
        self.sigma_range = (min_sigma,max_sigma)
        self.dim = dim
        self.channels = channels

    def __call__(x):
        kernel = self.gaussian_kernel()
        kernel_size = 2*self.size + 1
    
        x = x[None,...]
        padding = int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        x = torch.squeeze(F.conv2d(x, kernel, groups=3))
    
        return x
    
    def gaussian_kernel(self):
        """
        The gaussian kernel is the product of the 
        gaussian function of each dimension.
        """
        # unpacking
        size = self.size
        dim = self.dim
        channels = self.channels 
        sigma = th_uniform(*self.sigma_range,1)[0].to(tensor.device)

        # kernel_size should be an odd number.
        kernel_size = 2*size + 1
    
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32)
                                    for size in kernel_size])
        
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
    
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
    
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    
        return kernel



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
            y = torch.randint(h)
            x = torch.randint(w)
            y1 = torch.clamp(y - b_h // 2, 0, h)
            y2 = torch.clamp(y + b_h // 2, 0, h)
            x1 = torch.clamp(x - b_w // 2, 0, w)
            x2 = torch.clamp(x + b_w // 2, 0, w)            
            mask = torch.normal(self.mean,self.std,(y2-y1,x2-x1))
            mask = torch.Tensor(mask).to(pic.device)
            mask = torch.clamp(mask,0,1.)
            pic_n[y1:y2, x1:x2] = mask
            pics.append(pic_n)
        return pics
