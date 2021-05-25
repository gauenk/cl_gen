"""
ECCV 2020

"""

# -- python imports --
import pdb,pickle,lmdb,glob
from PIL import Image
from functools import partial
from easydict import EasyDict as edict
import numpy.random as npr
from pathlib import Path
import numpy as np
from einops import rearrange, repeat, reduce
import xml.etree.ElementTree as ET

# -- pytorch imports --
import torch,torchvision
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision import transforms as th_transforms
from torchvision.transforms import functional as tvxF
from torch.utils.data.distributed import DistributedSampler

# -- project imports --
from settings import ROOT_PATH
from datasets.transforms import TransformsSimCLR,AddGaussianNoiseSet,ScaleZeroMean,AddGaussianNoiseSetN2N,GaussianBlur,AddGaussianNoiseRandStd,GlobalCameraMotionTransform,AddGaussianNoise,AddPoissonNoiseBW,AddLowLightNoiseBW
from .common import get_loader


def get_rebel2021_dataset(cfg,mode):
    root = Path(ROOT_PATH)/ Path("./data/") /Path("rebel2021/")
    data = edict()
    batch_size = cfg.batch_size
    # create_foreground_images("./data/sun2009/")
    if mode == 'default':
        dynamic_info = edict()
        dynamic_info.num_frames = cfg.N
        data = edict()
        data.tr = Rebel2021(root)
        data.val,data.te = data.tr,data.tr
    elif mode == 'dynamic':
        dynamic_info = edict()
        dynamic_info.num_frames = cfg.N
        data = edict()
        data.tr = Rebel2021Dynamic(root)
        data.val,data.te = data.tr,data.tr
    else: raise ValueError(f"Unknown Rebel2021 mode {mode}")
    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader

class Rebel2021():

    @staticmethod
    def shuffle(filenames):
        D = len(filenames)
        torch.randperm(D)
        indices = torch.randperm(D)
        filenames = [filenames[index] for index in indices]
        return filenames


    def __init__(self,root_path,bw=True):

        self.root_path = Path(root_path)
        self.image_path = self.root_path / Path("images128")
        self.spoof = torch.Tensor([0.])
        self.bw = bw


        # -- load filenames --
        search_str = str(self.image_path / "./crop_*JPG")
        self.filenames = []
        for image_fn in glob.glob(search_str):
            self.filenames.append(image_fn)
        self.filenames = self.shuffle(self.filenames)
        D = len(self.filenames)

    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self,index):

        # -- create images --
        noisy_image = Image.open( str(self.filenames[index]) )
        if self.bw: noisy_image = noisy_image.convert("L")
        noisy_image = np.array(noisy_image , dtype=np.float32)
        if self.bw: noisy_image = np.repeat(noisy_image[...,None],3,-1)
        spoof_res,spoof_dir = self.spoof,self.spoof
        noisy_image = rearrange(noisy_image,'h w c -> c h w') / 255. - 0.5
        noisy_image = torch.tensor( noisy_image ).unsqueeze(0)
        spoof_clean = noisy_image.clone()
        return noisy_image, spoof_res, spoof_clean, spoof_dir

class DynamicRebel2021(Rebel2021):

    def __init__( self, root_path, bw = True ):
        super(DynamicRebel2021, self).__init__( root_path, bw = bw )
        self.dynamic_info = dynamic_info
        self.size = self.dynamic_info.frame_size
        self.image_set = image_set

    def __getitem__(self,index):

        # -- create images --
        noisy_image = Image.open( str(self.filenames[index]) )
        if self.bw: noisy_image = noisy_image.convert("L")
        noisy_image = np.array(noisy_image , dtype=np.float32)
        if self.bw: noisy_image = np.repeat(noisy_image[...,None],3,-1)
        spoof_res,spoof_dir = self.spoof,self.spoof
        noisy_image = rearrange(noisy_image,'h w c -> c h w') / 255. - 0.5
        noisy_image = torch.tensor( noisy_image ).unsqueeze(0)
        spoof_clean = noisy_image.clone()
        return noisy_image, spoof_res, spoof_clean, spoof_dir



def prepare_rebel2021_lmdb(cfg,dataset,ds_split,epochs=1,maxNumFrames=10,numSim=8,patchsize=3,id_str="all"):

    # -- check some configs --
    assert cfg.noise_type == 'g', "Noise must be Gaussian here"
    assert np.isclose(cfg.noise_params['g']['stddev'],25.), "Noise Level must be 25."
    assert cfg.N == maxNumFrames, "Dataset must load max number of frames."
    assert cfg.dynamic.ppf == 1, "Movement must be one pixel per frame."
    assert cfg.batch_size == 1, "Batch Size must be 1"

    # -- get noise level --
    noise_level = 0
    if cfg.noise_type == 'g': noise_level = int(cfg.noise_params['g']['stddev'])
    else: raise ValueError(f"Unknown Noise Type [{cfg.noise_type}]")
    
    # -- configs --
    num_images = epochs*len(dataset)

    # -- create target path --
    lower_name = cfg.dataset.name.lower()
    ds_path = Path(cfg.dataset.root) / lower_name / Path("./lmdbs")
    if not ds_path.exists(): ds_path.mkdir(parents=True)

    # -- create config strings --
    noise_str = "{}{}".format(cfg.noise_type,noise_level)
    sim_shuffle = "randPerm"
    nf_str = "nf{}".format(maxNumFrames)

    # -- create file names --

    # -- old --
    # lmdb_path = ds_path / Path("./noisy_burst_{}_{}_nf10_ns8_ps3_ppf1_fs128_{}_lmdb_{}".format(ds_split,noise_str,sim_shuffle,id_str))
    # metadata_fn = lmdb_path / Path("./noisy_burst_{}_{}_nf10_ns8_ps3_ppf1_fs128_{}_metadata_{}.pkl".format(ds_split,noise_str,sim_shuffle,id_str))
    lmdb_path = ds_path / Path("./noisy_burst_xburst_{}_{}_{}_ns8_ps3_ppf1_fs128_{}_lmdb_{}".format(ds_split,noise_str,nf_str,sim_shuffle,id_str))
    metadata_fn = lmdb_path / Path("./noisy_burst_xburst_{}_{}_{}_ns8_ps3_ppf1_fs128_{}_metadata_{}.pkl".format(ds_split,noise_str,nf_str,sim_shuffle,id_str))
    if lmdb_path.exists() and not overwite_lmdb(lmdb_path,metadata_fn): return
        

    # -- compute bytes per entry --
    burst, res_imgs, raw_img, directions = dataset[0]
    burst = burst.to(cfg.gpuid)
    sim_burst = compute_similar_bursts(cfg,burst.unsqueeze(1),numSim,patchsize=3,shuffle_k=True,pick_2=True)
    burst_nbytes = burst.cpu().numpy().astype(np.float32).nbytes
    simburst_nbytes = sim_burst.cpu().numpy().astype(np.float32).nbytes
    rawimg_nbytes = raw_img.numpy().astype(np.float32).nbytes
    dirc_nbytes = directions.numpy().astype(np.float32).nbytes
    data_size = (burst_nbytes + simburst_nbytes + rawimg_nbytes + dirc_nbytes) * num_images
    data_mb,data_gb = data_size/(1000.**2.),data_size/(1000.**3)
    print( "%2.2f MB | %2.2f GB" % (data_mb,data_gb) )

    # -- open lmdb file & open writer --
    print(f"Writing LMDB Path: {lmdb_path}")
    env = lmdb.open( str(lmdb_path) , map_size=data_size*1.5)
    txn = env.begin(write=True)

    # -- start lmdb writing loop --
    lmdb_index = 0
    tqdm_iter = tqdm(enumerate(range(num_images)), total=num_images, leave=False)
    commit_interval = 3 

    # -- load cached randperms --
    kindex_ds = kIndexPermLMDB(cfg.batch_size,maxNumFrames)

    for index, key in tqdm_iter:

        # -- write update --
        assert index == key, "These are not the same?"
        tqdm_iter.set_description('Write {}'.format(key))

        # -- load sample --
        burst, res_imgs, raw_img, directions = dataset[index]
        burst = burst.to(cfg.gpuid)
        kindex = kindex_ds[index].to(cfg.gpuid)
        sim_burst = compute_similar_bursts(cfg,burst.unsqueeze(1),numSim,
                                           patchsize=3,shuffle_k=True,
                                           kindex=kindex,pick_2=True)
        burst = burst.cpu()
        sim_burst = sim_burst.cpu()

        # -- sample to numpy --
        burst = burst.numpy()
        raw_img = raw_img.numpy()
        directions = directions.numpy()
        sim_burst = sim_burst.numpy()
        
        # -- create keys for lmdb --
        key_burst = "{}_burst".format(lmdb_index).encode('ascii')
        key_sim_burst = "{}_sim_burst".format(lmdb_index).encode('ascii')
        key_raw = "{}_raw".format(lmdb_index).encode('ascii')
        key_direction = "{}_direction".format(lmdb_index).encode('ascii')
        lmdb_index += 1
        
        # -- add to buffer to write for lmdb --
        txn.put(key_burst, burst)
        txn.put(key_sim_burst, sim_burst)
        txn.put(key_raw, raw_img)
        txn.put(key_direction, directions)

        # -- write to lmdb --
        if (index + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    # -- final write to lmdb & close --
    txn.commit()
    env.close()
    print('Finish writing lmdb')

    # -- write meta info to pkl --
    meta_info = {"num_samples": num_images,
                 "num_sim": numSim,
                 "num_frames":maxNumFrames,
                 "patch_size": patchsize}
    pickle.dump(meta_info, open(metadata_fn, "wb"))

    # -- done! --
    print('Finish creating lmdb meta info.')
    

    
    
    
