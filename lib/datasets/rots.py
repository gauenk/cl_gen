"""
Some rotation clips

"""

# -- python imports --
import re
import glob
from pathlib import Path
import PIL
from PIL import Image
from easydict import EasyDict as edict
import numpy as np

# -- pytorch imports --
import torch
from torchvision import transforms as tvT

# -- project imports --
from datasets.transforms import get_noise_transform

# -- [local] project imports --
from .common import get_loader

class Rots():

    def __init__(self,root,noise_info,n_frames=4,frame_size=256,frame_skip=0):
        # -- setup path --
        self.burst_path = root
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.skip_mod = frame_skip + 1

        # -- setup noise --
        self.noise_params = noise_info[noise_info.ntype]
        self.noise_trans = get_noise_transform(noise_info)

        # -- setup "clean" transform --
        self.clean_xform = tvT.Compose([tvT.ToTensor()])

        # -- load burst directories --
        burst_dirs = {}
        for burst_dir in glob.glob(str(self.burst_path / "./*")):
            burst_dir = Path(burst_dir).stem
            if burst_dir == "raw": continue
            burst_id = re.match(".*rot(?P<id>[0-9]+).*",burst_dir).groupdict()['id']
            burst_dirs[burst_id] = burst_dir
        self.burst_dirs = burst_dirs
        
    def __len__(self):
        return len(self.burst_dirs)

    def _load_image_filenames(self,image_dir):
        full_dir_search = self.burst_path / Path(image_dir) / Path("./*png")
        img_fns = {}
        for img_fn in glob.glob(str(full_dir_search)):
            img_id = re.match(".*_(?P<id>[0-9]+)",Path(img_fn).stem).groupdict()['id']
            if (int(img_id)-1) % self.skip_mod != 0: continue
            img_id_skip = str((int(img_id)-1)//self.skip_mod)
            img_fns[img_id_skip] = img_fn
        return img_fns

    def _open_burst(self,image_files):
        # -- open images --
        x,y,fs = 825,85,self.frame_size
        crop_box = [x,y,x+fs,y+fs]
        N = np.min([len(image_files),self.n_frames])        
        burst = [None for n in range(N)]
        for img_id,img_fn in image_files.items():
            if int(img_id) >= N: continue
            image = Image.open(img_fn).convert("RGB")
            image = image.crop(crop_box)
            burst[int(img_id)] = image
        return burst

    def __getitem__(self,index):

        # -- load burst with index --
        burst_dir = self.burst_dirs[str(index+1)]
        image_files = self._load_image_filenames(burst_dir)
        burst = self._open_burst(image_files)

        # -- format burst for method --
        N = np.min([len(burst),self.n_frames])
        noisy_burst = []
        for n in range(N):
            # -- add noise to noisy burst --
            noisy_frame = self.noise_trans(burst[n])
            noisy_burst.append(noisy_frame)

            # -- convert pils to image --
            burst[n] = self.clean_xform(burst[n])

        noisy_burst = torch.stack(noisy_burst,dim=0).unsqueeze(1)
        burst = torch.stack(burst,dim=0).unsqueeze(1)
        res_burst = noisy_burst - burst + 0.5

        print(f"N: {N}")
        # -- compat with package --
        spoof_dir = torch.tensor([0.])

        # -- return values --
        rinfo = {}
        rinfo['burst'] = noisy_burst
        rinfo['res'] = res_burst
        rinfo['clean_burst'] = burst
        rinfo['clean'] = burst[N//2]
        rinfo['directions'] = spoof_dir

        return rinfo

def get_rots_dataset(cfg,mode):
    root = cfg.dataset.root
    root = Path(root)/Path("rots")
    data = edict()
    rtype = 'dict' if cfg.dataset.dict_loader else 'list'
    if mode in ["default","dynamic"]:
        skip = 0 if not hasattr(cfg,'rot') or not hasattr(cfg.rot,'skip') else cfg.rot.skip
        batch_size = cfg.batch_size
        data.tr = Rots(root,cfg.noise_params,cfg.N,cfg.frame_size,skip)
        data.val = data.tr
        data.te = data.tr
    else: raise ValueError(f"Unknown ROTS dataset mode [{mode}]")
    
    loader = get_loader(cfg,data,batch_size,mode)
    return data,loader
