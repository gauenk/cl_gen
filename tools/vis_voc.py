
# python imports
import sys
sys.path.append("./lib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from easydict import EasyDict as edict
import numpy as np
from PIL import Image

# pytorch imports
import torch
import torch.nn as nn
import torchvision.utils as vutils

# project imports
import settings
from datasets import get_dataset
from n2n.config import get_cfg,get_args
from datasets.transform import GlobalCameraMotionTransform
from pyutils.timer import Timer

def main():

    args = get_args()
    cfg = get_cfg(args)

    cfg.S = 50000
    cfg.N = 30
    cfg.noise_type = 'g'
    cfg.noise_params['g']['stddev'] = 50
    cfg.dataset.name = "voc"
    cfg.dynamic = edict()
    cfg.dynamic.bool = True
    cfg.dynamic.ppf = 2
    cfg.dynamic.frames = cfg.N
    cfg.dynamic.mode = "global"
    cfg.dynamic.global_mode = "shift"
    cfg.dynamic.frame_size = 128
    cfg.dynamic.total_pixels = 20
    cfg.use_ddp = False
    cfg.use_collate = True
    cfg.set_worker_seed = False
    cfg.batch_size = 8
    cfg.num_workers = 4

    data,loader = get_dataset(cfg,'dynamic')
    noisy_trans = data.tr._get_noise_transform(cfg.noise_type,cfg.noise_params[cfg.noise_type])
    motion = GlobalCameraMotionTransform(cfg.dynamic,noisy_trans,True)
    noisy_l,rec_l,raw_l = [],[],[]
    timer = Timer()
    timer.tic()
    for index in range(cfg.batch_size):
        img = Image.open(data.tr.images[index])
        noisy,rec,raw = motion(img)
        noisy_l.append(noisy),rec_l.append(rec),raw_l.append(raw)
    timer.toc()
    print(timer)
    noisy = torch.stack(noisy_l,dim=1)
    rec = torch.stack(rec_l,dim=1)
    raw = torch.stack(raw_l)
    # noisy,raw = next(iter(loader.tr))
    noisy += 0.5
    noisy.clamp_(0,1.)

    rec += 0.5
    rec.clamp_(0,1.)

    raw = raw.expand(noisy.shape)
    images = torch.cat([noisy,rec,raw],dim=1)

    fig,ax = plt.subplots(figsize=(10,10))
    grids = [vutils.make_grid(images[i],nrow=8) for i in range(cfg.dynamic.frames)]
    ims = [[ax.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in grids]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)
    path = f"{settings.ROOT_PATH}/test_voc.mp4"
    ani.save(path, writer=writer)
    print(f"Wrote to {path}")
    
if __name__ == "__main__":
    main()
