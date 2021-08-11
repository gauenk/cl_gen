
# -- python --
import copy
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- pytorch --
import torch
import torchvision.transforms as tvT

# -- project --
from pyutils import print_tensor_stats
from datasets import load_dataset
from datasets.common import get_loader
from datasets.transforms import get_noise_transform,get_dynamic_transform

def load_image_dataset(cfg):

    # -- sims from cfg --
    noise_fxn,dynamic_fxn = transforms_from_cfg(cfg)

    # -- cfg copy without noise --
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.noise_params.ntype = 'none'
    cfg_copy.noise_params['none'] = {}
    data,loader = load_dataset(cfg_copy,'denoising')

    # -- wrapped --
    wdata = edict({})
    for key,value in data.items():
        wdata[key] = WrapperDataset(value,noise_fxn,dynamic_fxn)
    wloader = get_loader(cfg,wdata,cfg.batch_size,None)
    return wdata,wloader

def load_resample_dataset(cfg,records):

    # -- sims from cfg --
    noise_fxn,dynamic_fxn = transforms_from_cfg(cfg)

    # -- cfg copy without noise --
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.noise_params.ntype = 'none'
    cfg_copy.noise_params['none'] = {}
    data,loader = load_dataset(cfg_copy,'denoising')

    # -- wrapped --
    wdata = edict({})
    for key,value in data.items():
        wrapper = WrapperDataset(value,noise_fxn,dynamic_fxn)
        wdata[key] = ResampleWrapperDataset(wrapper,records[key])
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
            burst = results[0]+0.5
            flow = results[3]
            tl_index = results[4]
            return burst,flow,tl_index
        return wrapped
    def nonoise(image): return image

    # -- dynamics --
    dynamic_info = cfg.dynamic_info
    dynamic_raw_xform = get_dynamic_transform(dynamic_info,nonoise)
    dynamic_xform = dynamic_wrapper(dynamic_raw_xform)
    
    return noise_xform,dynamic_xform

class WrapperDataset():

    def __init__(self,data,noise_fxn,dynamic_fxn):
        self.data = data
        self.noise_fxn = noise_fxn
        self.dynamic_fxn = dynamic_fxn
        self.FULL_IMAGE_INDEX = 2 # from dataset object "data" such as PascalVoc
        self.cuda = True

    def __len__(self):
        return len(self.data)
    
    def _set_random_state(self,rng_state):
        torch.set_rng_state(rng_state['th'])
        np.random.set_state(rng_state['np'])

    def _get_random_state(self):
        th_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()
        rng_state = edict({'th':th_rng_state,'np':np_rng_state})
        return rng_state

    def __getitem__(self,index):

        # -- get random state --
        rng_state = self._get_random_state()

        # -- get image + dynamics + noise --
        image = self.data[index][self.FULL_IMAGE_INDEX]
        burst,flow,tl_index = self.dynamic_fxn(image)
        noisy = self.noise_fxn(burst+0.5)

        # -- auxillary vars --
        T = burst.shape[0]
        sburst = repeat(burst[T//2],'c h w -> t c h w',t=T)
        snoisy = self.noise_fxn(sburst)
        index_th = torch.IntTensor([index])

        sample = {'burst':burst,'noisy':noisy,
                  'flow':flow,'index':index_th,
                  'sburst':sburst,'snoisy':snoisy,
                  'rng_state':rng_state,
                  'tl_index':tl_index}
        return sample

    def sample_from_info(self,info):

        # -- get current random state --
        rng_state = self._get_random_state()
        
        # -- set info random state --
        self._set_random_state(info['rng_state'])

        # -- get sample --
        sample = self.__getitem__(info['image_index'])

        # -- restore original random state --
        self._set_random_state(rng_state)

        return sample

class ResampleWrapperDataset():

    def __init__(self,wrapper_data,records):
        self.wrapper_dataset = wrapper_data
        self.records = records

    def __len__(self):
        return len(self.records['image_index'])
    
    def _set_random_state(self,rng_state):
        torch.set_rng_state(rng_state['th'])
        np.random.set_state(rng_state['np'])

    def _get_random_state(self):
        th_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()
        rng_state = edict({'th':th_rng_state,'np':np_rng_state})
        return rng_state

    def __getitem__(self,index):

        # -- info from index --
        info = {field:self.records[field][index] for field in self.records.keys()}

        # -- get current random state --
        rng_state = self._get_random_state()
        
        # -- set info random state --
        self._set_random_state(info['rng_state'])

        # -- get sample --
        sample = self.wrapper_dataset[info['image_index']]

        # -- restore original random state --
        self._set_random_state(rng_state)

        return sample


        

def sample_to_cuda(sample):
    for key in sample.keys():
        if torch.is_tensor(sample[key]):
            sample[key] = sample[key].cuda(non_blocking=True)

def dict_to_device(sample,device):
    for key in sample.keys():
        if torch.is_tensor(sample[key]):
            sample[key] = sample[key].to(device,non_blocking=True)


