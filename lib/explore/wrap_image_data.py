
# -- python --
import numpy as np
from easydict import EasyDict as edict

# -- pytorch --
import torchvision.transforms as tvT

# -- project --
from datasets import load_dataset
from datasets.common import get_loader
from datasets.transforms import get_noise_transform,get_dynamic_transform

def load_image_dataset(cfg):
    noise_fxn,dynamic_fxn = transforms_from_cfg(cfg)
    data,loader = load_dataset(cfg,'denoising')
    # -- wrapped --
    wdata = edict({})
    for key,value in data.items():
        wdata[key] = WrapperDataset(value,noise_fxn,dynamic_fxn)
    wloader = get_loader(cfg,wdata,cfg.batch_size,None)
    return wdata,wloader

def transforms_from_cfg(cfg):

    # -- noise transform --
    noise_xform = get_noise_transform(cfg.noise_params,use_to_tensor=False)

    # -- simple functions for compat. --
    def dynamic_wrapper(dynamic_raw_xform):
        def wrapped(image):
            pil_image = tvT.ToPILImage()(image).convert("RGB")
            results = dynamic_raw_xform(pil_image)
            burst = results[0]
            directions = results[3]
            return burst,directions
        return wrapped
    def nonoise(image): return image

    # -- dynamics --
    dynamic_info = cfg.dynamic
    dynamic_raw_xform = get_dynamic_transform(dynamic_info,nonoise)
    dynamic_xform = dynamic_wrapper(dynamic_raw_xform)
    
    return noise_xform,dynamic_xform

class WrapperDataset():

    def __init__(self,data,noise_fxn,dynamic_fxn):
        self.data = data
        self.noise_fxn = noise_fxn
        self.dynamic_fxn = dynamic_fxn
        self.FULL_IMAGE_INDEX = 2

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        image = self.data[index][self.FULL_IMAGE_INDEX]
        burst,directions = self.dynamic_fxn(image)
        noisy = self.noise_fxn(burst)
        return {'burst':burst,'noisy':noisy,'directions':directions}
