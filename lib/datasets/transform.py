"""
Thanks to Spijkervet for this code
"""

# python imports
import numpy as np
from joblib import Parallel, delayed
from functools import partial

# pytorch imports
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms as thT
import torchvision.transforms.functional as tvF

# project imports
from pyutils.timer import Timer

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
        self.std = std / 255.
        
    def __call__(self, tensor):
        pic = torch.normal(tensor.add(self.mean),self.std)
        return pic
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianNoiseRandStd(object):
    def __init__(self, mean=0., min_std=0,max_std=50):
        self.mean = mean
        self.min_std = min_std
        self.max_std = max_std
        
    def __call__(self, tensor):
        std = th_uniform(self.min_std,self.max_std,1)
        pic = torch.normal(tensor.add(self.mean),std)
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


class GlobalCameraMotionTransform():
    """
    Global camera motion.

    direction: the vector of where we currently go
    delta: the amount of time spent in a current direction

    -- MISC thinking --

    types of global camera motion
    "wiggle": random jittering around the central location
    - high number of direction changes
    "shift": moving in a fixed direction
    - low number of direction changes
    """

    def __init__(self,dynamic,noise_trans=None,load_res=False):
        self.dynamic = dynamic
        self.load_res = load_res
        self.nframes = dynamic['frames']
        self.ppf = dynamic['ppf']
        self.total_pixels = dynamic['total_pixels']
        self.PI = 2*torch.acos(torch.zeros(1)).item() 
        self.frame_size = self.dynamic.frame_size
        self.img_size = 256
        self.to_tensor = thT.Compose([thT.ToTensor()])
        self.szm = thT.Compose([ScaleZeroMean()])
        self.noise_trans = noise_trans
        self.reset_seed = False
        if "reset_seed" in list(dynamic.keys()):
            self.reset_seed = dynamic.reset_seed

    def __call__(self, pic):
        
        # pics = pic.unsqueeze(0).repeat(cfg.nframes,1,1,1) 
        clean_target = None
        middle_index = self.nframes // 2
        w,h = pic.size
        d = self.sample_direction()
        tl = self.init_coordinate(d,h,w)

        out_frame_size = (self.frame_size,self.frame_size)
        # tl_init = tl.clone()

        # -- compute ppf rate given fixed frames --
        if self.total_pixels > 0:
            raw_ppf = float(self.total_pixels) / (self.nframes-1)
        else:
            raw_ppf = self.ppf

        # -- compute pixels per frame and resize image for fractions -- 
        if raw_ppf < 1 and raw_ppf > 0:
            h_new,w_new = int(h/raw_ppf)+1,int(w/raw_ppf)+1
            tl = torch.IntTensor([int(x.item()/raw_ppf) for x in tl])
            # print(h_new,w_new,h,w)
            pic = tvF.resize(pic,(h_new,w_new))
            crop_frame_size = int(self.frame_size/raw_ppf)+1
            ppf = 1
        else:
            ppf = raw_ppf
            crop_frame_size = self.frame_size
        # print(f"ppf: {ppf}")

        # -- create list of indices -- 
        tl_list = [tl.clone()]
        for i in range(self.nframes-1):
            step = (torch.round((i+1) * d * ppf)).type(torch.int)
            tl_i = tl + step
            tl_list.append(tl_i)
        # a = tl_list[0].type(torch.float)
        # b = tl_list[-1].type(torch.float)
        # print("pix diff",torch.sqrt(torch.sum(( a - b)**2)).item())
        # print(tl_list[0],tl_list[-1],d,w,h,pic.size)

        # -- get clean image --
        w_new,h_new = pic.size
        tl_mid = tl_list[middle_index]
        t,l = tl_mid[0].item(),tl_mid[1].item()
        target = tvF.resized_crop(pic,t,l,crop_frame_size,crop_frame_size,out_frame_size)
        clean_target = self.to_tensor(target)
        
        # -- create noisy frames -- 
        create_frames = partial(self._crop_image,pic,tl_list,crop_frame_size,out_frame_size)
        if self.nframes <= 30:
            pics = []
            res = []
            for i in range(self.nframes):
                pic_i,res_i = create_frames(i)
                pics.append(pic_i),res.append(res_i)
        # print(torch.norm(tl.type(torch.FloatTensor) - tl_init.type(torch.FloatTensor)))
        else:
            nj = np.min([self.nframes // 5,8])
            both = Parallel(n_jobs=nj)(delayed(create_frames)(i) for i in range(self.nframes))
            pics = [x[0] for x in both]
            res = [x[1] for x in both]            
        pics = torch.stack(pics)
        res = torch.stack(res)
        # print(clean_target.min(),clean_target.max(),clean_target.mean())
        return pics,res,clean_target

    def _crop_image(self,pic,tl_list,crop_frame_size,out_frame_size,i):
        tl = tl_list[i]
        # print(torch.norm(tl.type(torch.FloatTensor) - tl_init.type(torch.FloatTensor)))
        t,l = tl[0].item(),tl[1].item()
        # print(t,l,t+self.frame_size,l+self.frame_size,h,w)             
        pic_i = tvF.resized_crop(pic,t,l,crop_frame_size,crop_frame_size,out_frame_size)
        res_i = torch.empty(0)
        if (not self.noise_trans is None):
            noisy_pic_i = self.noise_trans(pic_i)
            if self.load_res:
                pic_nmlz = self.szm(self.to_tensor(pic_i))
                res_i = noisy_pic_i - pic_nmlz
            pic_i = noisy_pic_i
        else:
            pic_i = self.szm(self.to_tensor(pic_i))
        return pic_i,res_i

    def sample_direction(self):
        if self.reset_seed:
            torch.manual_seed(0)
        # r = torch.sqrt(torch.rand(1))
        r = 1
        rand_int = torch.rand(1)
        # print("rand_int",rand_int)
        theta = rand_int * 2 * self.PI
        direction = torch.FloatTensor([r * torch.cos(theta),r * torch.sin(theta)])
        # direction = torch.FloatTensor([1.,0.])
        return direction

    def init_coordinate(self,direction,h,w):
        
        odd = torch.prod(direction).item() > 0
        quandrant = 0
        if odd:
            if torch.all(direction > 0):
                quandrant = 1
            else:
                quandrant = 3
        elif not odd:
            if direction[1] > 0:
                quandrant = 2
            else:
                quandrant = 4

        init = [-1,-1] # top-left corner
        if quandrant == 1:
            init = [0,0]
        elif quandrant == 2:
            init = [h - self.frame_size, 0]# bottom-left to top-left
        elif quandrant == 3:
            init = [h - self.frame_size, w - self.frame_size] # bottom-right to top-left
        elif quandrant == 4:
            init = [0,w - self.frame_size] # top-right to top-left
        else:
            raise ValueError("What happened here?")
        # print(direction,odd,quandrant)

        return torch.IntTensor(init)


# class GlobalCameraMotionTransform():
#     """
#     Global camera motion.

#     direction: the vector of where we currently go
#     delta: the amount of time spent in a current direction

#     -- MISC thinking --

#     types of global camera motion
#     "wiggle": random jittering around the central location
#     - high number of direction changes
#     "shift": moving in a fixed direction
#     - low number of direction changes
#     """

#     def __init__(self,dynamic,noise_trans=None):
#         self.dynamic = dynamic
#         self.nframes = dynamic['frames']
#         self.ppf = dynamic['ppf']
#         self.total_pixels = dynamic['total_pixels']
#         self.PI = 2*torch.acos(torch.zeros(1)).item() 
#         self.frame_size = self.dynamic.frame_size
#         self.img_size = 256
#         self.to_tensor = thT.Compose([thT.ToTensor()])
#         self.szm = thT.Compose([ScaleZeroMean()])
#         self.noise_trans = noise_trans

#     def __call__(self, pic):
        
#         # pics = pic.unsqueeze(0).repeat(cfg.nframes,1,1,1) 
#         clean_target = None
#         middle_index = self.nframes // 2
#         h,w = pic.size
#         d = self.sample_direction()
#         tl = self.init_coordinate(d,h,w)
#         out_frame_size = (self.frame_size,self.frame_size)
#         pics = []
#         # tl_init = tl.clone()

#         if self.total_pixels > 0:
#             raw_ppf = float(self.total_pixels) / self.nframes
#         else:
#             raw_ppf = self.ppf

#         if raw_ppf < 1:
#             h_new,w_new = int(h/raw_ppf)+1,int(w/raw_ppf)+1
#             tl = torch.IntTensor([int(x.item()/raw_ppf) for x in tl])
#             # print(h_new,w_new,h,w)
#             pic = tvF.resize(pic,(h_new,w_new))
#             crop_frame_size = int(self.frame_size/raw_ppf)+1
#             ppf = 1
#         else:
#             ppf = raw_ppf
#             crop_frame_size = self.frame_size

#         # print(f"ppf: {ppf}")
#         for i in range(self.nframes):
#             step = (torch.round(d * ppf)).type(torch.int)
#             tl += step
#             # print(torch.norm(tl.type(torch.FloatTensor) - tl_init.type(torch.FloatTensor)))
#             t,l = tl[0].item(),tl[1].item()
#             # print(t,l,t+self.frame_size,l+self.frame_size,h,w)            
#             pic_i = tvF.resized_crop(pic,l,t,crop_frame_size,crop_frame_size,out_frame_size)
#             if (middle_index == i):
#                 clean_target = self.to_tensor(pic_i)
#             if (not self.noise_trans is None):
#                 pic_i = self.noise_trans(pic_i)
#             else:
#                 pic_i = self.szm(self.to_tensor(pic_i))
                
#             pics.append(pic_i)
#         # print(torch.norm(tl.type(torch.FloatTensor) - tl_init.type(torch.FloatTensor)))

#         pics = torch.stack(pics)
#         # print(clean_target.min(),clean_target.max(),clean_target.mean())
#         return pics,clean_target

#     def sample_direction(self):
#         # torch.manual_seed(0)
#         # r = torch.sqrt(torch.rand(1))
#         r = 1
#         rand_int = torch.rand(1)
#         # print("rand_int",rand_int)
#         theta = rand_int * 2 * self.PI
#         direction = torch.FloatTensor([r * torch.cos(theta),r * torch.sin(theta)])
#         # direction = torch.FloatTensor([1.,0.])
#         return direction

#     def init_coordinate(self,direction,h,w):
        
#         odd = torch.prod(direction).item() > 0
#         quandrant = 0
#         if odd:
#             if torch.all(direction > 0):
#                 quandrant = 1
#             else:
#                 quandrant = 3
#         elif not odd:
#             if direction[1] > 0:
#                 quandrant = 2
#             else:
#                 quandrant = 4

#         init = [-1,-1] # top-left corner
#         if quandrant == 1:
#             init = [0,0]
#         elif quandrant == 2:
#             init = [h - self.frame_size, 0]# bottom-left to top-left
#         elif quandrant == 3:
#             init = [h - self.frame_size, w - self.frame_size] # bottom-right to top-left
#         elif quandrant == 4:
#             init = [0,w - self.frame_size] # top-right to top-left
#         else:
#             raise ValueError("What happened here?")
#         # print(direction,odd,quandrant)

#         return torch.IntTensor(init)

