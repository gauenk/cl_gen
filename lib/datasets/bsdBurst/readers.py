
# -- python imports --
import os,tqdm
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange

# -- pytorch imports --
import torch
import torchvision

# -- project imports --
from align.nnf import compute_nnf

# --------------------------------------
#
#     Burst Reader (Paths and Images)
#
# --------------------------------------

def read_bsdBurst_paths(path):

    # -- get folders --
    spath = str(path)
    folders = [os.path.join(spath, o) for o in os.listdir(spath)
               if os.path.isdir(os.path.join(spath,o))]

    # -- create filenames --
    burst_paths = {}
    for folder in folders:
        folder = Path(folder)
        fname = folder.stem
        burst_paths[fname] = {}
        burst_paths[fname]['image'] = []
        burst_paths[fname]['nnf'] = []
        burst_paths[fname]['ref'] = -1
        for i in range(0,10):

            # -- read image --
            fpath0 = folder / f"{i}.jpg"
            fpath1 = folder / f"{i}.JPG"
            if fpath0.exists():
                burst_paths[fname]['image'].append(fpath0)
            elif fpath1.exists():
                burst_paths[fname]['image'].append(fpath1)
            else:
                raise ValueError(f"Uknown correct extention for {folder}")

            # -- read nnf info --
            nnf_path = folder / f"nnf_{i}_"
            burst_paths[fname]['nnf'].append(nnf_path)
                
        burst_paths[fname]['ref'] = len(burst_paths[fname]['image'])//2.

    # -- create deterministic order of folders --
    names = np.sort(list(burst_paths.keys()))
    
    return burst_paths,names

def read_bsdBurst_burst(paths,frame_size,nframes):

    # -- get indexing for paths --
    ref = int(paths['ref'])
    left = ref-nframes//2
    right = ref+nframes//2+1
    pslice = slice(left,right)

    # -- read images --
    burst = []
    img_paths = paths['image'][pslice]
    cc = torchvision.transforms.functional.center_crop
    for img_path in img_paths:
        img = Image.open(str(img_path)).convert('RGB')
        img = torch.FloatTensor(np.array(img))/255.
        img = rearrange(img,'h w c -> c h w')
        if not(frame_size is None):
            img = cc(img,frame_size)
        burst.append(img)
    burst = torch.stack(burst)        

    # -- get string of framesize --
    if not(frame_size is None):
        f0,f1 = frame_size
        fsize = f"{f0}_{f1}"

    # -- read/create nnf --
    nnf_vals,nnf_locs = [],[]
    nnf_paths = paths['nnf'][pslice]
    # for nnf_path in tqdm.tqdm(nnf_paths):
    for nnf_path in nnf_paths:

        # -- get image index fron nnf prefix --
        img_index = int(nnf_path.stem.split("_")[1])

        # -- read or create --
        if not(frame_size is None):
            vpath = Path(str(nnf_path) + f"vals_{fsize}.th")
            lpath = Path(str(nnf_path) + f"locs_{fsize}.th")
        else:
            vpath = Path(str(nnf_path) + f"vals.th")
            lpath = Path(str(nnf_path) + f"locs.th")

        # -- clean start 
        clean = False
        if clean:
            if vpath.exists(): os.remove(str(vpath))
            if lpath.exists(): os.remove(str(lpath))

        # -- read or create --
        if vpath.exists() and lpath.exists():
            vals = torch.load(str(vpath))
            locs = torch.load(str(lpath))
        else:
            # -- create nnf --
            ref_img = burst[ref]
            tgt_img = burst[img_index]
            vals,locs = compute_nnf(ref_img,tgt_img,3,K=3)

            # -- save nnf --
            torch.save(vals,str(vpath))
            torch.save(locs,str(lpath))
        nnf_vals.append(vals[0,0])
        nnf_locs.append(locs[0,0])

    data = {'burst':burst,'nnf_vals':nnf_vals,'nnf_locs':nnf_locs}
    return data
