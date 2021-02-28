"""
SUN2009 dataset

"""

# python imports
import pdb,glob
from PIL import Image
from functools import partial
from easydict import EasyDict as edict
import numpy.random as npr
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

# pytorch imports
import torch,torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as th_transforms
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import DatasetFolder

# project imports
from settings import ROOT_PATH
from pyutils.misc import add_noise
from .transform import TransformsSimCLR,AddGaussianNoiseSet,ScaleZeroMean,AddGaussianNoiseSetN2N,GaussianBlur,AddGaussianNoiseRandStd,GlobalCameraMotionTransform,AddGaussianNoise

class DynamicSUN2009():

    def __init__(self,path,N,noise_type,noise_params,dynamic):

        # -- create dynamic bsd68 --
        self.path = path
        self.image_path = path / Path("./images")
        self.label_path = path / Path("./labels")
        self.N = N
        self.noise_type = noise_type
        self.noise_params = noise_params
        self.dynamic = dynamic
        self.size = self.dynamic.frame_size

        # -- load data info --
        self.images,self.image_filenames = self._load_images(self.image_path)
        self.labels,self.label_filenames = self._load_labels(self.label_path)

        # -- augmentations --
        noisy_N = N if dynamic['bool'] else 1
        noisy_trans = self._get_noise_transform(noise_type,noise_params)
        self.dynamic_trans = self._get_dynamic_transform(dynamic,noisy_trans,True)
        th_trans = self._get_th_img_trans()
        self.th_trans = th_trans


    def _load_labels(self,label_path):
        labels,filenames = [],[]
        # -- load list of files --
        search_path = label_path / Path("*.regions.txt")
        for filename in glob.glob(str(search_path)):
            filenames.append(filename)            
        filenames = sorted(filenames)

        # -- load all labels --
        for filename in filenames:
            label = np.loadtxt(filename,delimiter=' ',dtype=np.int)
            labels.append(label)
        return labels,filenames

    def _load_images(self,image_path):
        images,filenames = [],[]
        # -- load list of files --
        search_path = image_path / Path("*.jpg")
        for filename in glob.glob(str(search_path)):
            filenames.append(filename)            
        filenames = sorted(filenames)

        # -- load all images --
        for filename in filenames:
            image = Image.open(filename).convert("RGB")
            images.append(image)
        return images,filenames

    def __len__(self):
        return len(self.image_filenames)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img = Image.open(self.images[index]).convert("RGB")
        img = self.images[index]
        # th_img = self.to_tensor(img)
        img_set,res_set,clean_target,direction = self.dynamic_trans(img)
        return img_set, res_set, clean_target, direction

    def _apply_transform_N(self,img):
        t = self.transform
        imgs = []
        for _ in range(self.N):
            # imgs.append(t_img)
            imgs.append(t(img))
        imgs = torch.stack(imgs)
        return imgs

    def _get_dynamic_transform(self,dynamic,noisy_trans,load_res=False):
        if dynamic['bool'] == False: 
            raise ValueError("We must set dynamics = True for the dynamic dataset loader.")
        if dynamic['mode'] == 'global':
            return GlobalCameraMotionTransform(dynamic,noisy_trans,load_res)
        elif dynamic['mode'] == 'local':
            raise NotImplemented("No local motion coded.")
        else:
            raise ValueError("Dynamic model [{dynamic['mode']}] not found.")

    def _get_noise_transform(self,noise_type,params):
        if noise_type == "g":
            return self._get_g_noise(params)
        elif noise_type == "ll":
            return self._get_ll_noise(params)
        elif noise_type == "msg":
            print("Loading msg transforms")
            return self._get_msg_noise(params)
        elif noise_type == "msg_simcl":
            print("Loading msg_simcl transforms")
            return self._get_msg_simcl_noise(params)
        else:
            raise ValueError(f"Unknown noise_type [{noise_type}]")

    def _get_g_noise(self,params):
        """
        Noise Type: Gaussian  (LL)
        - Each N images has Gaussian noise from with same parameters
        """
        to_tensor = th_transforms.ToTensor()
        szm = ScaleZeroMean()
        gaussian_noise = AddGaussianNoise(params['mean'],params['stddev'])
        comp = [to_tensor,szm,gaussian_noise]
        t = th_transforms.Compose(comp)
        return t

    def _get_ll_noise(self,params,N):
        """
        Noise Type: Low-Light  (LL)
        - Each N images is a low-light image with same alpha parameter
        """
        raise NotImplemented()

    def _get_msg_noise(self,params,N):
        """
        Noise Type: Multi-scale Gaussian  (MSG)
        - Each N images has it's own noise level
        """
        resize = torchvision.transforms.Resize(size=32)
        to_tensor = th_transforms.ToTensor()
        szm = ScaleZeroMean()
        gaussian_n2n = AddGaussianNoiseSetN2N(N,(0,50.))
        comp = [resize,to_tensor,szm,gaussian_n2n]
        t = th_transforms.Compose(comp)
        return t

    def _get_msg_simcl_noise(self,params,N):
        """
        Noise Type: Multi-scale Gaussian  (MSG)
        - Each N images has it's own noise level

        plus contrastive learning augs
        - random crop (flip and resize)
        - color distortion
        - gaussian blur
        """
        comp = []
        # -- random resize, crop, and flip --
        crop = th_transforms.RandomResizedCrop((self.size,self.size))
        comp += [crop]

        # -- flipping --
        # vflip = torchvision.transforms.RandomVerticalFlip(p=0.5)
        # comp += [vflip]
        hflip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        comp += [hflip]

        # -- color jitter -- 
        s = params['s'] 
        c_jitter_kwargs = {'brightness':0.8*s,
                           'contrast':0.8*s,
                           'saturation':0.8*s,
                           'hue': 0.2*s}
        cjitter = torchvision.transforms.ColorJitter(**c_jitter_kwargs)
        cjitter = torchvision.transforms.RandomApply([cjitter], p=0.8)
        comp += [cjitter]

        # -- convert to gray --
        # gray = torchvision.transforms.RandomGrayscale(p=0.8)
        # comp += [gray]
        
        # -- gaussian blur --
        # gblur = GaussianBlur(size=3)
        # comp += [gblur]

        # -- convert to tensor --
        to_tensor = th_transforms.ToTensor()
        comp += [to_tensor]

        # -- center to zero mean, all within [-1,1] --
        # szm = ScaleZeroMean()
        # comp += [szm]

        # -- additive gaussian noise --
        # gaussian_n2n = AddGaussianNoiseRandStd(0,0,50)
        # comp += [gaussian_n2n]

        t = th_transforms.Compose(comp)

        # def t_n_raw(t,N,img):
        #     imgs = []
        #     for _ in range(N):
        #         imgs.append(t(img))
        #     imgs = torch.stack(imgs)
        #     return imgs
        # t_n = partial(t_n_raw,t,N)
        return t
        
    def _get_th_img_trans(self):
        t = th_transforms.Compose([th_transforms.RandomResizedCrop((self.size,self.size)),
                                   th_transforms.ToTensor()
        ])
        return t



#
# Loading the datasets in a project
#

def get_sun2009_dataset(cfg,mode):
    root = cfg.dataset.root
    root = Path(root)/Path("./sun2009/")
    data = edict()
    if mode == 'cl':
        batch_size = cfg.cl.batch_size
        low_light = cfg.cl.dataset.transforms.low_light
        data.tr = ClSUN2009(root,cfg.cl.image_size,train=True,low_light=low_light)
        data.val = ClSUN2009(root,cfg.cl.image_size,train=True,low_light=low_light)
        data.te = ClSUN2009(root,cfg.cl.image_size,train=False,low_light=low_light)
    elif mode == "simcl" or mode == "denoising":
        batch_size = cfg.batch_size
        N = cfg.N
        load_res = cfg.dataset.load_residual
        noise_type = cfg.noise_type
        noise_params = cfg.noise_params[noise_type]
        dynamic = cfg.dynamic
        data.tr = DenoiseSUN2009(root,N,noise_type,noise_params,dynamic,load_res,train=True)
        data.val = DenoiseSUN2009(root,N,noise_type,noise_params,dynamic,load_res,train=False)
        data.val.data = data.val.data[0:2*2048]
        data.val.targets = data.val.targets[0:2*2048]
        data.te = DenoiseSUN2009(root,N,noise_type,noise_params,dynamic,load_res,train=False)
    elif mode == "dynamic":
        batch_size = cfg.batch_size
        N = cfg.N
        load_res = cfg.dataset.load_residual
        noise_type = cfg.noise_type
        noise_params = cfg.noise_params[noise_type]
        dynamic = cfg.dynamic
        bw = cfg.dataset.bw
        data.tr = DynamicSUN2009(root,N,noise_type,noise_params,dynamic)
        data.val,data.te = data.tr,data.tr
    else: raise ValueError(f"Unknown SUN2009 mode {mode}")

    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#                 Common Functions
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def get_loader(cfg,data,batch_size,mode):
    if cfg.use_ddp:
        loader = get_loader_ddp(cfg,data)
    else:
        loader = get_loader_serial(cfg,data,batch_size,mode)
    return loader
    
def collate_fn(batch):
    noisy,clean = zip(*batch)
    noisy = torch.stack(noisy,dim=1)
    clean = torch.stack(clean,dim=0)
    return noisy,clean

def collate_triplet_fn(batch):
    noisy,res,clean,directions = zip(*batch)
    noisy = torch.stack(noisy,dim=1)
    res = torch.stack(res,dim=1)
    clean = torch.stack(clean,dim=0)
    directions = torch.stack(directions,dim=0)
    return noisy,res,clean,directions

def set_torch_seed(worker_id):
    torch.manual_seed(0)

def get_loader_serial(cfg,data,batch_size,mode):
    loader_kwargs = {'batch_size': batch_size,
                     'shuffle':True,
                     'drop_last':True,
                     'num_workers':cfg.num_workers,
                     'pin_memory':True}
    if cfg.set_worker_seed:
        loader_kwargs['worker_init_fn'] = set_torch_seed
    if cfg.use_collate:
        if cfg.dataset.triplet_loader:
            loader_kwargs['collate_fn'] = collate_triplet_fn
        else:
            loader_kwargs['collate_fn'] = collate_fn

    loader = edict()
    loader.tr = DataLoader(data.tr,**loader_kwargs)
    loader.val = DataLoader(data.val,**loader_kwargs)
    loader.te = DataLoader(data.te,**loader_kwargs)
    return loader

def get_loader_ddp(cfg,data):
    loader = edict()
    loader_kwargs = {'batch_size':cfg.batch_size,
                     'shuffle':False,
                     'drop_last':True,
                     'num_workers': 1, #cfg.num_workers,
                     'pin_memory':True}
    if cfg.use_collate:
        loader_kwargs['collate_fn'] = collate_fn

    ws = cfg.world_size
    r = cfg.rank
    loader = edict()

    sampler = DistributedSampler(data.tr,num_replicas=ws,rank=r)
    loader_kwargs['sampler'] = sampler
    loader.tr = DataLoader(data.tr,**loader_kwargs)

    del loader_kwargs['sampler']
    loader_kwargs['drop_last'] = False

    # sampler = DistributedSampler(data.val,num_replicas=ws,rank=r)
    # loader_kwargs['sampler'] = sampler
    loader.val = DataLoader(data.val,**loader_kwargs)

    # sampler = DistributedSampler(data.te,num_replicas=ws,rank=r)
    # loader_kwargs['drop_last'] = False
    # loader_kwargs['sampler'] = sampler
    loader.te = DataLoader(data.te,**loader_kwargs)

    return loader

