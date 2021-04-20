"""
Pascal voc dataset

"""

# python imports
import pdb,pickle,lmdb
from PIL import Image
from functools import partial
from easydict import EasyDict as edict
import numpy.random as npr
from pathlib import Path
import numpy as np
from einops import rearrange, repeat, reduce
import xml.etree.ElementTree as ET

# pytorch imports
import torch,torchvision
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision import transforms as tvT
from torchvision.transforms import functional as tvxF
from torch.utils.data.distributed import DistributedSampler

# project imports
from settings import ROOT_PATH
from pyutils.misc import add_noise
from pyutils.timer import Timer
from datasets.transforms import get_dynamic_transform,get_noise_transform
from .common import get_loader

class DenoiseVOC(VOCDetection):

    def __init__(self,root, N, noise_info, image_set, rtype='list'):

        # -- correctly super with new name for path considerations --
        self.__class__.__name__ = "voc"
        super(DenoiseVOC, self).__init__( root, image_set=image_set,transform=None)

        # -- set params --
        self.N = N
        self.noise_params = noise_info
        self.spoof = torch.Tensor([0.])

        # -- noise transform --
        noise_trans = get_noise_transform(noise_info)

        # -- create transform of raw image --
        raw_trans = [tvT.ToTensor()]
        raw_trans = tvT.Compose(raw_trans)

        # -- set the transforms --
        self.raw_trans = raw_trans
        self.noise_trans = noise_trans

        # -- return type [list,dict] --
        if not (rtype in ['list','dict']):
            raise ValueError(f"Uknown return type [{rtype}]")
        self._return_type = rtype

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(self.images[index]).convert("RGB")

        noisy_img = self.noise_trans(img)
        raw_img = self.raw_trans(img)

        spoof_res,spoof_dir = self.spoof,self.spoof
        if self._return_type == "list":
            return noisy_img, spoof_res, raw_img, spoof_dir
        elif self._return_type == "dict":
            return noisy_img, spoof_res, raw_img, spoof_dir
        else: raise ValueError("How did this happend? Invalid return type [{self._return_type}].")


class DynamicVOC(VOCDetection):

    def __init__(self,root,year,image_set,N,noise_info,dynamic_info,load_res,bw,rtype='list'):
        # -- super call with corrected paths --
        self.__class__.__name__ = "pascal_voc"
        if year == "2012":
            path = root / Path("./VOCdevkit/VOC2012/")
        elif year == "2007":
            path = root / Path("./VOCdevkit/VOC2007/")
        path = root
        super(DynamicVOC, self).__init__( path, year, image_set)

        # -- set init params --
        self.N = N
        self.noise_params = noise_info[noise_info.ntype]
        self.dynamic_info = dynamic_info
        self.size = self.dynamic_info.frame_size
        self.image_set = image_set
        self.bw = bw

        # -- return type --
        if not (rtype in ['list','dict']):
            raise ValueError(f"Uknown return type [{rtype}]")
        self._return_type = rtype

        # -- create transforms --
        noise_trans = get_noise_transform(noise_info,noise_only=True)
        self.dynamic_trans = get_dynamic_transform(dynamic_info,noise_trans,load_res)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img = Image.open(self.images[index]).convert("RGB")
        if self.bw: img = img.convert('1')
        img_set,res_set,clean_target,directions = self.dynamic_trans(img)

        if self._return_type == "list":
            return img_set, res_set, clean_target, directions
        elif self._return_type == "dict":
            return {'burst':img_set, 'res':res_set, 'clean':clean_target, 'directions':directions}
        else: raise ValueError("How did this happend? Invalid return type [{self._return_type}].")

class DynamicVOC_LMDB_All():

    def __init__(self,lmdb_path,metadata_fn,N,noise_type,noise_params,dynamic_info,rtype='list'):

        # -- init params --
        self.lmdb_path = lmdb_path
        self.N = N
        # self.noise_info = noise_info
        self.dynamic_info = dynamic_info
        self.size = self.dynamic_info.frame_size
        self.lmdb_fields = ['burst','sim_burst','raw','direction']
        self.fields = ['burst','sim_burst','raw','directions']

        # -- load metadata --
        self.meta_info = pickle.load(open(metadata_fn,'rb'))
        assert self.N <= self.meta_info['num_frames'], f"Cannot exceed {self.meta_info['num_frames']} frames"
        self.data_env = None

        print(f"DynamicVOC LMDB Path: [{lmdb_path}]")

        # -- return type [list,dict] --
        if not (rtype in ['list','dict']):
            raise ValueError(f"Uknown return type [{rtype}]")
        self._return_type = rtype
        
        # -- ask and replace --
        # keys = list(self.meta_info.keys())
        # if "noise_type" in keys and self.meta_info["noise_type"] != cfg.noise_type:
        #     print(f"Updating noise type from [{cfg.noise_type}] to [{key}]")
        #     cfg.noise_type = key
        # if "noise_level" in keys and self.meta_info["noise_type"] == "g":
        #     cfg.noise_params['g']['stddev']

    def __len__(self):
        return self.meta_info['num_samples']

    def _read_img_noise_lmdb(self, data_env, lmdb_index, dtype=np.float32, shape=(3,128,128)):

        # -- create lmdb keys --
        keys = {}
        for field in self.lmdb_fields: keys[field] = "{}_{}".format(lmdb_index,field).encode('ascii')

        # -- read from lmdb --
        buffs = {}
        with data_env.begin(write=False) as txn:
            for field,key in keys.items():
                buffs[field] = txn.get(key)
        
        # -- convert to ndarrays --
        data = {}
        for field,buf in buffs.items():
            data[field] = np.frombuffer(buf, dtype=dtype)#.copy()

        # -- reshaping --
        num_frames = self.meta_info['num_frames']
        for field,sample in data.items():
            if field == "burst": data[field] = sample.reshape(num_frames,*shape)
            if field == "sim_burst": data[field] = sample.reshape(2,num_frames,*shape)
            if field == "raw": data[field] = sample.reshape(shape)

        # -- adjust to target number of frames --
        Md,Mt = num_frames//2,self.N//2
        sN,eN = (Md - Mt),(Md - Mt)+self.N
        data['burst'] = data['burst'][sN:eN]
        data['sim_burst'] = rearrange(data['sim_burst'][:,sN:eN],'k n c h w -> n k c h w')

        # -- torch tensor --
        # for field,sample in data.items(): data[field] = torch.tensor(sample.copy())
        for field,sample in data.items(): data[field] = torch.tensor(sample.copy())
        data['directions'] = data['direction']
        del data['direction']
        return data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.data_env is None:
            self.data_env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
        data = self._read_img_noise_lmdb(self.data_env, index)

        # -- return for collate_fn --
        if self._return_type == 'list':
            ordered_data = [data[field] for field in self.fields]
            return ordered_data
        elif self._return_type == 'dict':
            data['clean'] = data['raw'] 
            del data['raw']
            return data
        else: raise ValueError("How did this happend? Invalid return type [{self._return_type}].")

class DynamicVOC_LMDB_Burst():

    def __init__(self,lmdb_path,metadata_fn,N,noise_type,noise_params,dynamic_info,rtype='list'):

        # -- init params --
        self.lmdb_path = lmdb_path
        self.N = N
        # self.noise_info = noise_info
        self.dynamic_info = dynamic_info
        self.size = self.dynamic_info.frame_size
        self.lmdb_fields = ['burst','raw','direction']

        # -- load metadata --
        self.meta_info = pickle.load(open(metadata_fn,'rb'))
        assert self.N <= self.meta_info['num_frames'], f"Cannot exceed {self.meta_info['num_frames']} frames"
        self.data_env = None

        print(f"DynamicVOC LMDB Path: [{lmdb_path}]")
        
        # -- return type [list,dict] --
        if not (rtype in ['list','dict']):
            raise ValueError(f"Uknown return type [{rtype}]")
        self._return_type = rtype

        # -- ask and replace --
        # keys = list(self.meta_info.keys())
        # if "noise_type" in keys and self.meta_info["noise_type"] != cfg.noise_type:
        #     print(f"Updating noise type from [{cfg.noise_type}] to [{key}]")
        #     cfg.noise_type = key
        # if "noise_level" in keys and self.meta_info["noise_type"] == "g":
        #     cfg.noise_params['g']['stddev']

    def __len__(self):
        return self.meta_info['num_samples']

    def _read_img_noise_lmdb(self, data_env, lmdb_index, dtype=np.float32, shape=(3,128,128)):

        # -- create lmdb keys --
        keys = {}
        for field in self.lmdb_fields: keys[field] = "{}_{}".format(lmdb_index,field).encode('ascii')

        # -- read from lmdb --
        t = Timer()
        t.tic()
        buffs = {}
        with data_env.begin(write=False) as txn:
            for field,key in keys.items():
                buffs[field] = txn.get(key)

        # -- convert to ndarrays --
        data = {}
        for field,buf in buffs.items():
            data[field] = np.frombuffer(buf, dtype=dtype).copy()

        # -- reshaping --
        num_frames = self.meta_info['num_frames']
        for field,sample in data.items():
            if field == "burst": data[field] = sample.reshape(num_frames,*shape)
            if field == "raw": data[field] = sample.reshape(shape)
            
        # -- adjust to target number of frames --
        Md,Mt = num_frames//2,self.N//2
        sN,eN = (Md - Mt),(Md - Mt)+self.N
        data['burst'] = data['burst'][sN:eN]

        # -- torch tensor --
        for field,sample in data.items(): data[field] = torch.tensor(sample)
        data['directions'] = data['direction']
        del data['direction']
        return data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.data_env is None:
            self.data_env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
        data = self._read_img_noise_lmdb(self.data_env, index)

        # -- return for collate_fn --
        if self._return_type == 'list':
            ordered_data = [data[field] for field in self.lmdb_fields]
            return ordered_data
        elif self._return_type == 'dict':
            data['clean'] = data['raw'] 
            del data['raw']
            return data
        else: raise ValueError("How did this happend? Invalid return type [{self._return_type}].")


#
# Loading the datasets in a project
#

def get_voc_dataset(cfg,mode):
    root = cfg.dataset.root
    root = Path(root)/Path("voc")
    data = edict()
    rtype = 'dict' if cfg.dataset.dict_loader else 'list'
    if mode == 'cl':
        batch_size = cfg.cl.batch_size
        low_light = cfg.cl.dataset.transforms.low_light
        data.tr = ClVOC(root,cfg.cl.image_size,train=True,low_light=low_light)
        data.val = ClVOC(root,cfg.cl.image_size,train=True,low_light=low_light)
        data.te = ClVOC(root,cfg.cl.image_size,train=False,low_light=low_light)
    elif mode == "simcl" or mode == "denoising":
        batch_size = cfg.batch_size
        N = cfg.N
        load_res = cfg.dataset.load_residual
        noise_type = cfg.noise_type
        noise_params = cfg.noise_params[noise_type]
        noise_info = cfg.noise_params
        dynamic = cfg.dynamic
        rtype = 'dict' if cfg.dataset.dict_loader else 'list'
        data.tr = DenoiseVOC(root,N,noise_info,image_set='train',rtype=rtype)
        data.val = DenoiseVOC(root,N,noise_info,image_set='train',rtype=rtype)
        # data.val.data = data.val.data[0:2*2048]
        # data.val.targets = data.val.targets[0:2*2048]
        data.te = DenoiseVOC(root,N,noise_info,image_set='val',rtype=rtype)
    elif mode == "dynamic":
        batch_size = cfg.batch_size
        N = cfg.N
        load_res = cfg.dataset.load_residual
        noise_type = cfg.noise_type
        noise_params = cfg.noise_params[noise_type]
        noise_info = cfg.noise_params
        dynamic_info = cfg.dynamic
        bw = cfg.dataset.bw
        data.tr = DynamicVOC(root,"2012","trainval",N,noise_info,dynamic_info,load_res,bw,rtype)
        D = -1
        if D > 0:
            print(f"Limiting Dataset Size to [{D}]")
            data.tr.images = data.tr.images[:D]
        data.val = DynamicVOC(root,"2012","val",N,noise_info,dynamic_info,load_res,bw,rtype)
        # data.val.data = data.val.data[0:2*2048]
        # data.val.targets = data.val.targets[0:2*2048]
        data.te = DynamicVOC(root,"2007","test",N,noise_info,dynamic_info,load_res,bw,rtype)

    elif mode == "dynamic-lmdb-all":
        batch_size = cfg.batch_size
        N = cfg.N
        load_res = cfg.dataset.load_residual
        noise_type = cfg.noise_type
        noise_params = cfg.noise_params[noise_type]
        dynamic = cfg.dynamic

        # -- create config strings --
        num_samples = 100
        id_str = "all"
        noise_level = int(cfg.noise_params['g']['stddev'])
        noise_str = "{}{}".format(cfg.noise_type,noise_level)
        sim_shuffle = "randPerm"
        frame_str = "nf{}".format(cfg.N)

        # -- create file names --
        stem = root / Path("./lmdbs")
        ds_split = "train"
        # lmdb_path = stem / Path("./noisy_burst_{}_{}_nf10_ns8_ps3_ppf1_fs128_{}_lmdb_{}".format(ds_split,noise_str,noise_str,sim_shuffle))
        # metadata_fn = lmdb_path / Path("./noisy_burst_{}_{}_nf10_ns8_ps3_ppf1_fs128_{}_metadata_{}.pkl".format(ds_split,noise_str,noise_str,sim_shuffle,id_str))
        
        # lmdb_path = stem / Path("./noisy_burst_{}_{}_nf10_ns8_ps3_ppf1_fs128_{}_lmdb_{}".format(ds_split,noise_str,sim_shuffle,id_str))
        # metadata_fn = lmdb_path / Path("./noisy_burst_{}_{}_nf10_ns8_ps3_ppf1_fs128_{}_metadata_{}.pkl".format(ds_split,noise_str,sim_shuffle,id_str))

        # -- all standard --
        lmdb_path = stem / Path("./noisy_burst_xburst_{}_{}_{}_ns8_ps3_ppf1_fs128_{}_lmdb_{}".format(ds_split,noise_str,frame_str,sim_shuffle,id_str))
        metadata_fn = lmdb_path / Path("./metadata.pkl")
        data.tr = DynamicVOC_LMDB_All(lmdb_path,metadata_fn,N,noise_type,noise_params,dynamic,rtype=rtype)
        data.val,data.te = data.tr,data.tr

    elif mode == "dynamic-lmdb-burst":
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # 
        #     Burst Only Standard 
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        batch_size = cfg.batch_size
        N = cfg.N
        load_res = cfg.dataset.load_residual
        noise_type = cfg.noise_type
        noise_params = cfg.noise_params[noise_type]
        dynamic = cfg.dynamic

        # -- config for str --
        num_samples = 100
        id_str = "all"
        noise_level = int(cfg.noise_params['g']['stddev'])
        noise_str = "{}{}".format(cfg.noise_type,noise_level)
        sim_shuffle = "randPerm"
        nf_str = "nf{}".format(8)

        # -- create file names --
        stem = root / Path("./lmdbs")
        ds_split = "train"
        id_str= "burst"
        size_str= "num{}".format(num_samples)
        ds_split = "train"
        lmdb_path = stem / Path("./noisy_burst_{}_{}_{}_{}_ppf1_fs128_lmdb_{}".format(ds_split,size_str,noise_str,nf_str,id_str))
        metadata_fn = lmdb_path / Path("./metadata.pkl")

        # -- load sets
        data.tr = DynamicVOC_LMDB_Burst(lmdb_path,metadata_fn,N,noise_type,noise_params,dynamic,rtype=rtype)
        data.val = data.tr

        ds_split = "test"
        lmdb_path = stem / Path("./noisy_burst_{}_{}_{}_{}_ppf1_fs128_lmdb_{}".format(ds_split,size_str,noise_str,nf_str,id_str))
        metadata_fn = lmdb_path / Path("./metadata.pkl")
        data.te = DynamicVOC_LMDB_Burst(lmdb_path,metadata_fn,N,noise_type,noise_params,dynamic,rtype=rtype)
        # print(f"RTYPE: {rtype}")

    else: raise ValueError(f"Unknown VOC mode {mode}")

    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader

def get_voc_transforms(cfg):
    cls_batch_size = cfg.cls.batch_size
    cfg.cls.batch_size = 1
    data,loader = get_voc_dataset(cfg,'cls')
    cfg.cls.batch_size = cls_batch_size

    transforms = edict()
    transforms.tr = get_dataset_transforms(cfg,data.tr.data)
    transforms.te = get_dataset_transforms(cfg,data.te.data)

    return transforms

def get_dataset_transforms(cfg,data):
    import numpy as np
    noise_levels = cfg.imgrec.dataset.noise_levels
    noise_data = []
    for noise_level in noise_levels:
        shape = (len(data),3,33,33)
        means = torch.zeros(shape)
        noise_i = torch.normal(means,noise_level)
        noise_data.append(noise_i)
    noise_data = torch.stack(noise_data,dim=1)
    return noise_data

