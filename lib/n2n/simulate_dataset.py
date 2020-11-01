"""

1. repeat the random noise patters periodically
2. change the order ofthe random noise patters
3. change the size of the dataset

"""


# python imports
from tqdm import tqdm
import os,sys,uuid
sys.path.append("./lib")
from easydict import EasyDict as edict
from pathlib import Path
import numpy as np
import numpy.random as npr
from PIL import Image

# pytorch imports
import torch

# project imports
import settings
from datasets import load_dataset
from n2n.config import get_cfg,get_args
from datasets.cifar10 import DenoiseCIFAR10


# def simulate_noisy_dataset(cfg,data,loader):
#     sim_data,sim_loader = edict(),edict()
#     set_splits = ['tr']#list(data.keys())
#     for set_split in set_splits:
#         ds,l = data[set_split],loader[set_split]
#         data_split,loader_split = simulate_noisy_dataset_set(cfg,set_split,ds,l)
#         sim_data[set_split] = data_split
#         sim_loader[set_split] = loader_split
#     return sim_data,sim_loader

def generate_noisy_dataset(cfg,data,loader):
    # Ggrid = [10,25,50,100,150,200]
    # GSTRgrid = ['g10','g25','g50','g100','g150','g200']
    Ggrid = [10,50,150]
    GSTRgrid = ['g10','g50','g150']
    Ngrid = [30]
    tr_data = data.tr
    tr_loader = loader.tr
    for idxN,N in enumerate(Ngrid):
        for G,GSTR in zip(Ggrid,GSTRgrid):
            cfg.N = N
            cfg.noise_type = 'g'
            cfg.noise_params['g']['stddev'] = G
            path = noisy_dataset_filename(cfg,"train",GSTR,idxN)
            if path.exists(): continue
            sim_dataset = simulate_noisy_dataset(cfg,"train",tr_data,tr_loader)
            print(f"Writing sim dataset to path: {path}")
            torch.save(sim_dataset,path)
                
    
def simulate_noisy_dataset(cfg,set_split,dataset,loader):

    # -=-=- Unpacking -=-=-
    assert isinstance(dataset,DenoiseCIFAR10), "We need the denoising loader."
    data = dataset.data
    noise_params = cfg.noise_params['g']
    noisy_transform,repN = dataset._get_noise_transform(cfg.noise_type,noise_params,cfg.N)
    clean_transform = dataset._get_th_img_trans()
    
    # -=-=-=- Get root path -=-=-=-
    lower_dsname = cfg.dataset.name.lower()
    root = Path(f"{settings.ROOT_PATH}/data/{lower_dsname}/n2n/{set_split}/")
    if not root.exists(): root.mkdir(parents=True)
    
    # -=-=-=- Save randomized index order -=-=-=-
    random_indices = npr.permutation(len(data))
    clean_path = root / Path(f"rand_indices.pt")
    if clean_path.exists(): random_indices = torch.load(clean_path)
    else: torch.save(random_indices,clean_path)

    # -=-=-=- Randomly shuffle indices -=-=-=-
    data = data[random_indices]

    # -=-=-=- For each image, create N noisy samples -=-=-=-
    sim_dataset = []
    for image in tqdm(data):
        pil_image = Image.fromarray(image)
        noisy_images = noisy_transform(pil_image)
        clean_image = clean_transform(pil_image).unsqueeze(0)
        all_images = torch.cat([noisy_images,clean_image],dim=0)
        sim_dataset.append(all_images)
    sim_dataset = torch.stack(sim_dataset)
    return sim_dataset


def noisy_dataset_filename(cfg,set_split,noise_name,idxN):
    lower_dsname = cfg.dataset.name.lower()
    root = Path(f"{settings.ROOT_PATH}/data/{lower_dsname}/n2n/{set_split}/")
    if not root.exists(): root.mkdir(parents=True)
    fn = Path(f"{cfg.N}_{idxN}_{noise_name}.pt")
    path = root / fn
    return path
    
# def save_noisy_dataset(cfg,set_split,S_clean_images,S_noisy_images):
#     pass


def load_noisy_datsaet(cfg,path):
    pass
    

    # resize = torchvision.transforms.Resize(size=32)
    # to_tensor = th_transforms.ToTensor()
    # szm = ScaleZeroMean()
    # gaussian_noise = AddGaussianNoiseSet(N,params['mean'],params['stddev'])
    # comp = [resize,to_tensor,szm,gaussian_noise]
    # t = th_transforms.Compose(comp)
    
    # for sample in data:
    #     sample



def test_me():

    args = get_args()
    cfg = get_cfg(args)
    cfg.N = 2
    cfg.S = 100
    # load data
    data,loader = load_dataset(cfg,'denoising')
    generate_noisy_dataset(cfg,data,loader)
    # data,loader = simulate_noisy_dataset(cfg,data,loader)
    exit()
    
if __name__ == "__main__":
    print("HI")
    test_me()
